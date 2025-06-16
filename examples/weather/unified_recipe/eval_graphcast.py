import os
import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from graphcast_datapipes import SeqZarrDatapipe_GraphCast
from graphcast_reordering import *
from normalisation_wrapper import Norm_Wrapper_GraphCast
from utils import get_filesystem
from loss_weights import *
import gc
OmegaConf.register_new_resolver("eval", eval)
# from save_interactive_plots import create_interactive_forecast_comparison_html
import xarray as xr

from modulus import Module
from modulus.distributed import DistributedManager
from modulus.launch.logging import (
    LaunchLogger,
    PythonLogger,
)
from modulus.utils import StaticCaptureEvaluateNoGrad
import pandas as pd
from typing import Any

def create_forecast_dataset(
    observations: np.ndarray,
    base_times: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    variable_names: List[str],
    timestep_interval_hours: int,
    variable_attributes: Optional[Dict[str, Dict[str, Any]]] = None,
    input_timesteps: Optional[int] = 2,
) -> xr.Dataset:
    """
    Combines forecast tensors, start times and 

    Args:
        observations (np.ndarray): The observation data tensor with shape
                                   (n_base_times, n_timesteps, n_variables, n_lats, n_lons).
        base_times (np.ndarray): 1D array of forecast base times (start times).
                                 Should contain datetime-like objects (e.g., np.datetime64).
        lats (np.ndarray): 1D array of latitude values.
        lons (np.ndarray): 1D array of longitude values.
        variable_names (List[str]): List of variable names corresponding to the
                                    'n_variables' dimension of the observations tensor.
        timestep_interval_hours (int): The time difference between consecutive forecast
                                      timesteps in hours (e.g., 6).
        variable_attributes (Optional[Dict[str, Dict[str, Any]]]):
            An optional dictionary where keys are variable names and values are
            dictionaries of attributes (e.g., {'units': 'K', 'long_name': 'Air Temp'}).
        input_timesteps (Optional[int]): Number of timesteps taken by the model as input to shift
                                        the base_times array forward by
    Returns:
        xr.Dataset: An xarray Dataset containing the structured forecast data.

    Raises:
        ValueError: If dimension sizes inferred from observations do not match
                    the lengths of coordinate arrays or variable_names list.
    """

    try:
        n_base_times, n_timesteps, n_variables, n_lats, n_lons = observations.shape
    except ValueError as e:
        raise ValueError(f"Observations tensor does not have 5 dimensions. Shape: {observations.shape}") from e

    if n_base_times != len(base_times):
        raise ValueError(f"Observation dimension 0 size ({n_base_times}) doesn't match "
                         f"len(base_times) ({len(base_times)})")
    if n_variables != len(variable_names):
         raise ValueError(f"Observation dimension 2 size ({n_variables}) doesn't match "
                         f"len(variable_names) ({len(variable_names)})")
    if n_lats != len(lats):
         raise ValueError(f"Observation dimension 3 size ({n_lats}) doesn't match "
                         f"len(lats) ({len(lats)})")
    if n_lons != len(lons):
         raise ValueError(f"Observation dimension 4 size ({n_lons}) doesn't match "
                         f"len(lons) ({len(lons)})")

    if not np.issubdtype(base_times.dtype, np.datetime64):
        try:
            base_times = pd.to_datetime(base_times).to_numpy()
        except Exception as e:
            raise TypeError("Could not convert base_times to np.datetime64. "
                            f"Ensure they are datetime-like objects. Original error: {e}") from e


    steps = pd.to_timedelta(np.arange(n_timesteps) * timestep_interval_hours, unit='h')
    base_times = [x + pd.to_timedelta(input_timesteps*timestep_interval_hours, unit='h') for x in base_times]
    data_vars = {}
    for i, var_name in enumerate(variable_names):
        attrs = variable_attributes.get(var_name, {}) if variable_attributes else {}

        data_vars[var_name] = xr.DataArray(
            observations[:, :, i, :, :],
            coords=[base_times, steps, lats, lons],
            dims=['base_time', 'step', 'latitude', 'longitude'],
            name=var_name,
            attrs=attrs
        )

    ds = xr.Dataset(data_vars)

    ds['base_time'].attrs['long_name'] = 'Forecast Base Time'
    ds['step'].attrs['long_name'] = 'Forecast Lead Time (offset from base_time)'
    ds['latitude'].attrs['units'] = 'degrees_north'
    ds['latitude'].attrs['long_name'] = 'Latitude'
    ds['longitude'].attrs['units'] = 'degrees_east'
    ds['longitude'].attrs['long_name'] = 'Longitude'

    ds['valid_time'] = ds.base_time + ds.step
    ds['valid_time'].attrs['long_name'] = 'Forecast Valid Time'
    ds = ds.set_coords('valid_time')

    return ds

@hydra.main(version_base="1.2", config_path="conf", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    """
    Graphcast Evaluation Function
    """

    OmegaConf.resolve(cfg)
    os.makedirs(cfg.validation.save_dir, exist_ok=True)
    DistributedManager.initialize()
    dist = DistributedManager()

    LaunchLogger.initialize(use_mlflow=False)
    logger = PythonLogger("main")
    model = Module.instantiate(
        {
            "__name__": cfg.model.name,
            "__args__": {
                k: tuple(v) if isinstance(v, ListConfig) else v
                for k, v in cfg.model.args.items()
            }
        }
    )
    try:
        model.load(cfg.validation.eval_load_path)
        print(f"Loaded Model from {cfg.validation.eval_load_path}")
    except Exception as e:
        print(e)
        exit(1)
    device = dist.device
    model = model.to(device)
    
    fs = get_filesystem(
        cfg.filesystem.type,
        cfg.filesystem.key,
        cfg.filesystem.endpoint_url,
        cfg.filesystem.region_name,
    )

    val_dataset_mapper = fs.get_mapper(cfg.curated_dataset.val_dataset_filename)

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
    
    latitude = xr.open_zarr(cfg.curated_dataset.val_dataset_filename).coords['latitude'].values
    longitude = xr.open_zarr(cfg.curated_dataset.val_dataset_filename).coords['longitude'].values
    base_times = xr.open_zarr(cfg.curated_dataset.val_dataset_filename).coords['time'].values

    loss_weights = get_weights((cfg.model.nr_output_channels, cfg.model.input_shape[0], cfg.model.input_shape[1]), latitude, levels_by_order, per_variable_weight_mapping, longitude)
    loss_weights = loss_weights.to(device)
    criterion = WeightedMSELoss(loss_weights)
    model = Norm_Wrapper_GraphCast(model, input_std, input_mean, output_std, 
                                   ORIGINAL_ORDER_INPUTS_176, ORIGINAL_ORDER_OUTPUTS_83, 
                                   reorder_178_to_original_176, original_176_to_original_83, 
                                   reorder_178_to_original_output, reorder_output_to_original_output)
    permutation = reorder_output_to_original_output
    variable_order = ORIGINAL_ORDER_OUTPUTS_83
    labels = [x[0] for x in variable_order if x[0] in ['2m_temperature', 'total_precipitation']]
    relevant_indices = [idx for idx, x in enumerate(variable_order) if x[0] in ['2m_temperature', 'total_precipitation']]
    def unroll(model, constants, inputs, forcings, node_features, num_steps = 1):
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
            input = torch.concat((constants, forcings[i], model_pred_i_minus_1.squeeze(), forcings[i+1], model_pred_i_0.squeeze().to(constants.device)), dim=0)
            model_pred_i_0, norm_predictions = model(input, forcings[i+2], node_features)
            
            model_targets.append(inputs[i+2].unsqueeze(0))
            model_inputs.append(inputs[i+1].unsqueeze(0))
            model_predicted.append(model_pred_i_0)
            model_norm_predictions.append(norm_predictions)
            model_pred_i_0 = model_pred_i_0[..., original_output_to_reorder_output, :, :]
        
        model_predicted = torch.stack(model_predicted, dim=1)
        model_targets = torch.stack(model_targets, dim=1)
        model_inputs = torch.stack(model_inputs, dim=1)
        model_norm_predictions = torch.stack(model_norm_predictions, dim=1)
        return model_predicted, model_targets, model_inputs, model_norm_predictions
    
    @StaticCaptureEvaluateNoGrad(model=model, logger=logger, use_graphs=False)
    def eval_forward(model, constants, inputs, forcings, node_features, criterion, nr_steps, permutation = reorder_output_to_original_output):
        with torch.no_grad():
            outputs, targets, model_inputs, model_norm_predictions = unroll(
                model, constants, inputs, forcings, node_features, nr_steps
            )

            loss = model.loss(model_inputs, model_norm_predictions, targets, criterion)
        
        return loss, outputs, targets[..., permutation, :, :]
    # Wrap validation in launch logger for console / WandB logs
    predictions = []
    actuals = []
    times = []
    with LaunchLogger("valid", epoch=0) as log:
        model.model.eval()

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
            (
                loss,
                outputs,
                targets,
            ) = eval_forward(model, constants.squeeze()[0], 
                                inputs, forcings.squeeze(), node_features.squeeze()[0], 
                                criterion, cfg.validation.num_steps, permutation=permutation)
            predictions.append(outputs[..., relevant_indices, :, :].cpu().numpy())
            actuals.append(targets[..., relevant_indices, :, :].cpu().numpy())
            times.append(base_times[i])
            loss_epoch += loss.detach().cpu().numpy()
            num_examples += targets.shape[0]
            if i % 6 == 0:
                logger.info(i)
            # gc.collect()
            # torch.clear_autocast_cache()
            # torch.cuda.empty_cache()
        
        log.log_epoch({"Validation error": loss_epoch / num_examples})
        avg_val_loss = loss_epoch / num_examples
        log.log_epoch({"Validation error": avg_val_loss})
    predictions = np.stack(predictions, axis=0)
    actuals = np.stack(actuals, axis=0)
    times = np.array(times)
    prediction_xr = create_forecast_dataset(predictions.squeeze(), times, latitude, longitude, labels, 6, input_timesteps=2)
    actual_xr = create_forecast_dataset(actuals.squeeze(), times, latitude, longitude, labels, 6, input_timesteps=2)
    prediction_xr.to_zarr(os.path.join(cfg.validation.save_dir, ('predicted.zarr' if not cfg.validation.is_base else 'predicted_base.zarr')))
    actual_xr.to_zarr(os.path.join(cfg.validation.save_dir, 'groundtruth.zarr'))
if __name__ == "__main__":
    main()
