# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

import hydra
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import zarr
from omegaconf import DictConfig, ListConfig, OmegaConf
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
import torch.optim as torch_optimizers
from tqdm import tqdm
from graphcast_datapipes import SeqZarrDatapipe_GraphCast
from graphcast_reordering import *
from normalisation_wrapper import Norm_Wrapper_GraphCast
from utils import get_filesystem
from loss_weights import *
import datetime
# Add eval to OmegaConf TODO: Remove when OmegaConf is updated
OmegaConf.register_new_resolver("eval", eval)

import xarray as xr

from modulus import Module
from modulus.distributed import DistributedManager
from modulus.launch.logging import (
    LaunchLogger,
    PythonLogger,
    RankZeroLoggingWrapper,
)
from modulus.launch.logging.mlflow import initialize_mlflow
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.utils import StaticCaptureEvaluateNoGrad, StaticCaptureTraining

from model_packages import save_inference_model_package


def batch_normalized_mse(pred: Tensor, target: Tensor) -> Tensor:
    """Calculates batch-wise normalized mse error between two tensors."""

    pred_flat = pred.reshape(pred.size(0), -1)
    target_flat = target.reshape(target.size(0), -1)

    diff_norms = torch.linalg.norm(pred_flat - target_flat, ord=2.0, dim=1)
    target_norms = torch.linalg.norm(target_flat, ord=2.0, dim=1)

    error = diff_norms / target_norms
    return torch.mean(error)


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function for unified weather model training.
    """

    # Resolve config so that all values are concrete
    OmegaConf.resolve(cfg)

    # Initialize distributed environment for training
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize loggers
    # initialize_mlflow(
    #     experiment_name=cfg.experiment_name,
    #     experiment_desc=cfg.experiment_desc,
    #     run_name=f"{cfg.model.name}-trainng",
    #     run_desc=cfg.experiment_desc,
    #     user_name="Modulus User",
    #     mode="offline",
    # )
    LaunchLogger.initialize(use_mlflow=False)  # Modulus launch logger
    logger = PythonLogger("main")  # General python logger
    # Initialize model
    model = Module.instantiate(
        {
            "__name__": cfg.model.name,
            "__args__": {
                k: tuple(v) if isinstance(v, ListConfig) else v
                for k, v in cfg.model.args.items()
            },  # TODO: maybe mobe this conversion to resolver?
        }
    )
    try:
        model.load(cfg.model.pretrain_load_path)
        print(f"Loaded Model from {cfg.model.pretrain_load_path}")
    except Exception as e:
        print(e)
    device = dist.device
    # device = 'cpu'
    dummy_norm = torch.nn.BatchNorm2d(5)
    writer = None
    if dist.rank == 0:
        # Create a timestamped log directory
        current_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join("logs", current_time_str)
        os.makedirs(log_dir, exist_ok=True) # Ensure the directory exists
        writer = SummaryWriter(log_dir)
        logger.info(f"Tensorboard logs will be saved to: {log_dir}")
    if cfg.model.force_single_checkpoint:
        model.set_checkpoint_model(True)
    else:
        if cfg.model.checkpoint_encoder:
            model.set_checkpoint_encoder(True)
        if cfg.model.checkpoint_decoder:
            model.set_checkpoint_decoder(True)
        if cfg.model.checkpoint_processor_segments:
            model.set_checkpoint_processor(cfg.model.checkpoint_processor_segments)

    model = model.to(device)
    # Distributed learning
    if dist.world_size > 1:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],
                output_device=device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    optimizers = torch_optimizers
    OptimizerClass = getattr(optimizers, cfg.training.optimizer.name)
    optimizer = OptimizerClass(model.parameters(), **cfg.training.optimizer.args)

    # Attempt to load latest checkpoint if one exists
    if dist.world_size > 1:
        torch.distributed.barrier()
    # loaded_epoch = load_checkpoint(
    #     "./checkpoints",
    #     models=model,
    #     optimizer=optimizer,
    #     scheduler=None,
    #     device=device,
    # )
    loaded_epoch = 0
    # Initialize filesytem (TODO: Add multiple filesystem support)
    fs = get_filesystem(
        cfg.filesystem.type,
        cfg.filesystem.key,
        cfg.filesystem.endpoint_url,
        cfg.filesystem.region_name,
    )

    # Get filesystem mapper for datasets
    train_dataset_mapper = fs.get_mapper(cfg.curated_dataset.train_dataset_filename)
    val_dataset_mapper = fs.get_mapper(cfg.curated_dataset.val_dataset_filename)

    # Initialize validation datapipe
    val_datapipe = SeqZarrDatapipe_GraphCast(
        file_mapping=val_dataset_mapper,
        variable_groups=cfg.curated_dataset.variable_groups,
        batch_size=cfg.validation.batch_size,
        num_steps=cfg.validation.num_steps + cfg.model.nr_input_steps,
        shuffle=False,
        device=device,
        process_rank=dist.rank,
        world_size=dist.world_size,
        batch=cfg.datapipe.batch,
        parallel=cfg.datapipe.parallel,
        num_threads=cfg.datapipe.num_threads,
        prefetch_queue_depth=cfg.datapipe.prefetch_queue_depth,
        py_num_workers=cfg.datapipe.py_num_workers,
        py_start_method=cfg.datapipe.py_start_method,
    )

    levels_by_order = []
    try:
        variable_weights = cfg.variable_weights
    except:
        # Default Graphcast Config
        variable_weights = {
            "10m_u_component_of_wind": 0.1,
            "10m_v_component_of_wind": 0.1,
            "mean_sea_level_pressure": 0.1,
            "total_precipitation": 0.1,
        }
    per_variable_weight_mapping = {}
    for idx, variable in enumerate(ORIGINAL_ORDER_OUTPUTS_83):
        if isinstance(variable, str):
            name = variable
            levels_by_order.append(None)
        elif len(variable) == 2:
            name, _ = variable
            levels_by_order.append(None)
        else:
            name, _, level = variable
            levels_by_order.append(int(level[1]))
        if name in variable_weights.keys():
            per_variable_weight_mapping[idx] = variable_weights[name]
    input_mean = xr.load_dataset(cfg.model.input_mean).rename({'total_precipitation_6hr': 'total_precipitation'})
    input_std = xr.load_dataset(cfg.model.input_std).rename({'total_precipitation_6hr': 'total_precipitation'})
    output_std = xr.load_dataset(cfg.model.output_std).rename({'total_precipitation_6hr': 'total_precipitation'})
    
    latitude = xr.open_zarr(cfg.curated_dataset.train_dataset_filename).coords['latitude'].values
    longitude = xr.open_zarr(cfg.curated_dataset.train_dataset_filename).coords['longitude'].values

    loss_weights = get_weights((cfg.model.nr_output_channels, cfg.model.input_shape[0], cfg.model.input_shape[1]), latitude, levels_by_order, per_variable_weight_mapping, longitude)
    loss_weights = loss_weights.to(device)
    criterion = WeightedMSELoss(loss_weights)
    model = Norm_Wrapper_GraphCast(model, input_std, input_mean, output_std, 
                                   ORIGINAL_ORDER_INPUTS_176, ORIGINAL_ORDER_OUTPUTS_83, 
                                   reorder_178_to_original_176, original_176_to_original_83, 
                                   reorder_178_to_original_output, reorder_output_to_original_output)
    permutation = reorder_output_to_original_output
    variable_order = ORIGINAL_ORDER_OUTPUTS_83
    print("Unrolling Network")
    # Unroll network
    save_inference_model_package(
                    # model,
                    model.model,
                    cfg,
                    dummy_norm,
                    dummy_norm,
                    latitude=zarr.open(cfg.dataset.dataset_filename, mode="r")[
                        "latitude"
                    ],
                    longitude=zarr.open(cfg.dataset.dataset_filename, mode="r")[
                        "longitude"
                    ],
                    save_path="./model_package_{}".format(cfg.experiment_name),
                    readme="This is a model card for the global weather model.",
                )
    def unroll(model, constants, inputs, forcings, node_features, num_steps = 1):
        # Get number of steps to unroll
        possible_steps = min(inputs.shape[0], forcings.shape[0])
        if possible_steps < 3:
            raise ValueError("Need forcings at at least 3 different timesteps to make predictions")
        max_steps = possible_steps - 2
        model_pred_i_minus_1 = inputs[0]
        model_pred_i_0 = inputs[1]
        model_predicted = []
        model_targets = []
        model_inputs = []
        model_norm_predictions = []
        for i in range(min(num_steps, max_steps)):
            # print(constants.device, forcings[i].device, model_pred_i_minus_1.squeeze().device, forcings[i+1].device, model_pred_i_0.squeeze().device)
            # Create Input
            input = torch.concat((constants, forcings[i], model_pred_i_minus_1.squeeze(), forcings[i+1], model_pred_i_0.squeeze().to(constants.device)), dim=0)
            # input = input[..., reorder_178_to_original_176, :, :]
            # model_pred_i_minus_1 = model_pred_i_0
            # x = torch.concat((input, forcings[i+2], node_features), dim=-3).unsqueeze(0).to(device)
            # x = torch.rand_like(x)
            # model_pred_i_0 = model(x)
            # print(x.device, model_pred_i_0.device)
            model_pred_i_0, norm_predictions = model(input, forcings[i+2], node_features)
            
            model_targets.append(inputs[i+2].unsqueeze(0))
            model_inputs.append(inputs[i+1].unsqueeze(0))
            model_predicted.append(model_pred_i_0)
            model_norm_predictions.append(norm_predictions)
            # print(model_pred_i_0.device)
            model_pred_i_0 = model_pred_i_0[..., original_output_to_reorder_output, :, :]
            # print(model_pred_i_0.device)
        # Stack predictions
        
        model_predicted = torch.stack(model_predicted, dim=1)
        model_targets = torch.stack(model_targets, dim=1) # Currently reordered, but that is remedied in the loss_computation
        model_inputs = torch.stack(model_inputs, dim=1)
        model_norm_predictions = torch.stack(model_norm_predictions, dim=1)
        return model_predicted, model_targets, model_inputs, model_norm_predictions
    # Evaluation forward pass
    @StaticCaptureEvaluateNoGrad(model=model, logger=logger, use_graphs=False)
    def eval_forward(model, constants, inputs, forcings, node_features, criterion, nr_training_steps, permutation = reorder_output_to_original_output):
        # Forward pass
        with torch.no_grad():
            outputs, targets, model_inputs, model_norm_predictions = unroll(
                model, constants, inputs, forcings, node_features, nr_training_steps
            )

            # Get l2 loss
            # loss = criterion(outputs, targets.to(device))
            loss = model.loss(model_inputs, model_norm_predictions, targets, criterion)
        
        # Targets are re-ordered. To get same order as output, we need to permute to the original order
        return loss, outputs, targets[..., permutation, :, :]

    # Training forward pass


    @StaticCaptureTraining(
        model=model, optim=optimizer, logger=logger, use_amp=cfg.training.amp_supported
    )
    def train_step_forward(model, constants, inputs, forcings, node_features, criterion, nr_training_steps):
        # Handles forward and backward pass
        # Forward pass
        outputs, targets, model_inputs, model_norm_predictions = unroll(
            model, constants, inputs, forcings, node_features, nr_training_steps
        )
        # Get l2 loss, returning loss performs backward pass
        torch.cuda.empty_cache()
        # loss = criterion(outputs, targets.to(device))
        loss = model.loss(model_inputs, model_norm_predictions, targets, criterion)
        return loss


    # Main training loop
    global_epoch = 0
    total_steps_taken = 0
    for stage in cfg.training.stages:
        # print(f'Starting Stage {stage} of {cfg.training.stages}')
        stage_start_epoch = global_epoch
        current_stage_step = 0
        # Skip if loaded epoch is greater than current stage
        if loaded_epoch > global_epoch:
            # Check if current stage needs to be run
            if loaded_epoch >= global_epoch + stage.num_epochs:
                # Skip stage
                global_epoch += stage.num_epochs
                continue
            # Otherwise, run stage for remaining epochs
            else:
                num_epochs = stage.num_epochs - (loaded_epoch - global_epoch)
        else:
            num_epochs = stage.num_epochs

        # print(f"{num_epochs} epochs of {stage.num_epochs} done")
        
        # Create new datapipe
        train_datapipe = SeqZarrDatapipe_GraphCast(
            file_mapping=train_dataset_mapper,
            variable_groups=cfg.curated_dataset.variable_groups,
            batch_size=stage.batch_size,
            num_steps=stage.unroll_steps + cfg.model.nr_input_steps,
            shuffle=True,
            device=device,
            process_rank=dist.rank,
            world_size=dist.world_size,
            batch=cfg.datapipe.batch,
            parallel=cfg.datapipe.parallel,
            num_threads=cfg.datapipe.num_threads,
            prefetch_queue_depth=cfg.datapipe.prefetch_queue_depth,
            py_num_workers=cfg.datapipe.py_num_workers,
            py_start_method=cfg.datapipe.py_start_method,
        )

        # Initialize scheduler
        # if stage.lr_scheduler_name is not None:
        #     SchedulerClass = getattr(torch.optim.lr_scheduler, stage.lr_scheduler_name)
        #     scheduler = SchedulerClass(optimizer, **stage.args)

        #     # Set scheduler to current step
        #     scheduler.step(stage.num_epochs - num_epochs)
        # else:
        #     scheduler = None
        # Get current step for checking if max iterations is reached
        current_step = len(train_datapipe) * (stage.num_epochs - num_epochs)

        # Run number of epochs
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            current_epoch_global = stage_start_epoch + epoch
            # Wrap epoch in launch logger for console / WandB logs
            # print("Training")
            epoch_train_loss = 0.0
            steps_in_epoch = 0
            nr_bytes = 0
            with LaunchLogger(
                "train",
                epoch=epoch,
                num_mini_batch=len(train_datapipe),
                epoch_alert_freq=10,
            ) as log:
                # Track memory throughput
                tic = time.time()
                nr_bytes = 0
                # Training loop
                for _, data in enumerate(train_datapipe):
                    # Check if ran max iterations for stage
                    if current_step >= stage.max_iterations:
                        break
                    # Get predicted and unpredicted variables
                    constants = data[0]['constants']
                    inputs_surface = data[0]['inputs_surface']
                    inputs_pressure_levels = torch.reshape(data[0]['inputs_pressure_levels'], 
                                                           (1, 
                                                            cfg.model.nr_input_steps + stage.unroll_steps, 
                                                            (cfg.curated_dataset.nr_pressure_levels * 
                                                            cfg.curated_dataset.nr_inputs_pressure_levels), 
                                                            cfg.model.input_shape[0], cfg.model.input_shape[1]))
                    forcings = data[0]['forcings'].permute((0, 1, 2, 4, 3))
                    node_features = data[0]['node_features']
                    inputs = torch.concat((inputs_surface, inputs_pressure_levels), dim=-3).squeeze()
                    # Perform training step
                    loss = train_step_forward(model, constants.squeeze()[0], inputs, forcings.squeeze(), node_features.squeeze()[0], criterion, stage.unroll_steps)
                    epoch_train_loss += loss.item()
                    steps_in_epoch += 1
                    nr_bytes += inputs.element_size() * inputs.nelement()
                    nr_bytes += forcings.element_size() * forcings.nelement()
                    log.log_minibatch({"loss": loss.detach()})
                    if writer is not None:
                        writer.add_scalar('Loss/train_step', loss.item(), global_step=total_steps_taken)
                    # Increment current step
                    current_stage_step += 1
                    total_steps_taken += 1
                    current_step += 1
                # if stage.lr_scheduler_name is not None:
                #     scheduler.step()

                # Log learning rate
                log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
                avg_epoch_train_loss = epoch_train_loss / steps_in_epoch
                current_lr = optimizer.param_groups[0]['lr']
                throughput = nr_bytes / (time.time() - tic) / 1e9 if (time.time() - tic) > 0 else 0

                # Log memory throughput
                log.log_epoch({"GB/s": nr_bytes / (time.time() - tic) / 1e9})
                if writer is not None:
                    writer.add_scalar('Loss/train_epoch_avg', avg_epoch_train_loss, global_step=current_epoch_global)
                    writer.add_scalar('LearningRate', current_lr, global_step=current_epoch_global)
                    writer.add_scalar('Throughput/GB_per_sec', throughput, global_step=current_epoch_global)
                
            # print("Validating")
            # Perform validation
            if dist.rank == 0:
                # Wrap validation in launch logger for console / WandB logs
                with LaunchLogger("valid", epoch=epoch) as log:
                    # Switch to eval mode
                    # model.eval()
                    model.model.eval()

                    # Validation loop
                    loss_epoch = 0.0
                    num_examples = 0
                    for i, data in enumerate(val_datapipe):
                        constants = data[0]['constants']
                        inputs_surface = data[0]['inputs_surface']
                        inputs_pressure_levels = torch.reshape(data[0]['inputs_pressure_levels'], 
                                                               (cfg.validation.batch_size, cfg.model.nr_input_steps + 
                                                                cfg.validation.num_steps, 
                                                                (cfg.curated_dataset.nr_pressure_levels 
                                                                * cfg.curated_dataset.nr_inputs_pressure_levels), 
                                                                cfg.model.input_shape[0], cfg.model.input_shape[1]))
                        forcings = data[0]['forcings'].permute((0, 1, 2, 4, 3))
                        node_features = data[0]['node_features']
                        inputs = torch.concat((inputs_surface, inputs_pressure_levels), dim=-3).squeeze()
                        # Log memory throughput
                        # nr_bytes += (
                        #     predicted_variables.element_size()
                        #     * predicted_variables.nelement()
                        # )
                        # nr_bytes += (
                        #     unpredicted_variables.element_size()
                        #     * unpredicted_variables.nelement()
                        # )

                        # Increment current step
                        current_step += 1
                        (
                            loss,
                            outputs,
                            targets,
                        ) = eval_forward(model, constants.squeeze()[0], 
                                         inputs, forcings.squeeze(), node_features.squeeze()[0], 
                                         criterion, cfg.validation.num_steps, permutation=permutation)
                        
                        loss_epoch += loss.detach().cpu().numpy()
                        num_examples += targets.shape[0]
                        # Plot validation on first batch
                        if i == 0:
                            outputs = (
                                outputs.cpu().numpy()
                            )
                            targets = targets.cpu().numpy()
                            for chan in range(outputs.shape[-3]):
                                plt.close("all")
                                fig, ax = plt.subplots(
                                    3,
                                    outputs.shape[-4],
                                    figsize=(15, outputs.shape[-5] * 5),
                                    squeeze=False
                                )
                                for t in range(outputs.shape[-4]):
                                    ax[0, t].set_title(
                                        "Network prediction, Step {}".format(t)
                                    )
                                    ax[1, t].set_title(
                                        "Ground truth, Step {}".format(t)
                                    )
                                    ax[2, t].set_title("Difference, Step {}".format(t))
                                    ax[0, t].imshow(outputs[0, t, chan])
                                    ax[1, t].imshow(targets[0, t, chan])
                                    ax[2, t].imshow(
                                        outputs[0, t, chan]
                                        - targets[0, t, chan]
                                    )
                                    # if writer is not None:
                                    #     writer.add_figure(f"Validation/{variable_order[chan]}/{t}", fig, global_step = current_epoch_global)

                                fig.savefig(
                                    f"forecast_validation_var{variable_order[chan]}.png"
                                )

                    # Log validation loss
                    log.log_epoch({"Validation error": loss_epoch / num_examples})

                    # Switch back to train mode
                    # model.train()
                    model.model.train()
                    avg_val_loss = loss_epoch / num_examples
                    log.log_epoch({"Validation error": avg_val_loss})
                    logger.info(f"Epoch {current_epoch_global} Validation Loss: {avg_val_loss:.6f}")
                    if writer is not None:
                        writer.add_scalar('Loss/validation', avg_val_loss, global_step=current_epoch_global)
                    
            # Sync after each epoch
            if dist.world_size > 1:
                torch.distributed.barrier()

            # Save checkpoint
            if dist.rank == 0 and (epoch % 3 == 0): #  and
                # Use Modulus Launch checkpoint
                save_checkpoint(
                    "./checkpoints",
                    # models=[model],
                    models=[model.model],
                    optimizer=optimizer,
                    scheduler=None,
                    epoch=epoch,
                )

                # Save model package
                logger.info("Saving model card")
                save_inference_model_package(
                    # model,
                    model.model,
                    cfg,
                    dummy_norm,
                    dummy_norm,
                    latitude=zarr.open(cfg.dataset.dataset_filename, mode="r")[
                        "latitude"
                    ],
                    longitude=zarr.open(cfg.dataset.dataset_filename, mode="r")[
                        "longitude"
                    ],
                    save_path="./model_package_{}".format(cfg.experiment_name),
                    readme="This is a model card for the global weather model.",
                )

        # Finish training
        if dist.rank == 0:
            # Save model card
            logger.info("Finished training!")
    print("FINISHED")

if __name__ == "__main__":
    main()
