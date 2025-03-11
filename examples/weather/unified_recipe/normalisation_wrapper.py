from typing import Optional, Tuple, Union

import xarray
import torch
import logging


class InputsAndResiduals(torch.nn.Module):
    """Wraps a PyTorch model with a residual connection, normalizing inputs and target residuals.

    The inner PyTorch model is given inputs that are normalized using `locations`
    and `scales` to roughly zero-mean unit variance.

    For target variables that are present in the inputs, the inner model is
    trained to predict residuals (target - last_frame_of_input) that have been
    normalized using `residual_scales` (and optionally `residual_locations`) to
    roughly unit variance / zero mean.

    This replaces `residual.Predictor` in the case where you want normalization
    that's based on the scales of the residuals.

    Since we return the underlying model's loss on the normalized residuals,
    if the underlying model is a sum of per-variable losses, the normalization
    will affect the relative weighting of the per-variable loss terms (hopefully
    in a good way).

    For target variables *not* present in the inputs, the inner model is
    trained to predict targets directly, that have been normalized in the same
    way as the inputs.

    The transforms applied to the targets (the residual connection and the
    normalization) are applied in reverse to the predictions before returning
    them.
    
    Parameters
    ----------
    model : torch.nn.Module
        The neural network model used for prediction.
    stddev_by_level : torch.Tensor
        Standard deviation values for each (variable, level) combination in the dataset.
    mean_by_level : torch.Tensor
        Mean values for each for each (variable, level) combination in the dataset.
    diffs_stddev_by_level : torch.Tensor
        Standard deviation of differences between consecutive timesteps for each (variable, level) combination.
    input_variables : list[str]
        List of input variables used by the model. Precipitation absent for GC-Operational.
    output_variables : list[str]
        List of output variables predicted by the model.
    input_permutation : list[int]
        A permutation of input indices to rearrange input variables to expected model order.
    outputs_from_inputs : list[Optional[int]]
        Mapping from model input indices to output indices, specifying dependencies. Precipitation 'None' on GC-Operational
    outputs_from_full_inputs : list[Optional[int]]
        Mapping from all input indices (including auxiliary ones) to outputs. Precipitation present on GC-Operational
    """

    def __init__(
        self,
        model: torch.nn.Module,
        stddev_by_level: torch.tensor,
        mean_by_level: torch.tensor,
        diffs_stddev_by_level: torch.tensor,
        input_variables: list[str],
        output_variables: list[str],
        input_permutation: list[int],
        outputs_from_inputs: list[Union[Optional[int]]],
        outputs_from_full_inputs: list[Union[Optional[int]]]
    ):
        super().__init__()
        self.model = model
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.missing_variables = list(set(output_variables) - set(input_variables))

        self._scales = stddev_by_level
        self._locations = mean_by_level
        self._residual_scales = diffs_stddev_by_level
        self._residual_locations = None

        self.input_permutation = input_permutation
        self.outputs_from_input_order = outputs_from_inputs
        self.outputs_from_full_input_order = outputs_from_full_inputs

        self.input_scales = self._get_tensor_from_dataset(self.input_variables, [], stddev_by_level).view(len(self.input_variables), 1, 1)
        self.input_locations = self._get_tensor_from_dataset(self.input_variables, [], mean_by_level).view(len(self.input_variables), 1, 1)
        self.output_scales = self._get_tensor_from_dataset(self.output_variables, self.missing_variables, diffs_stddev_by_level, aux_dataset=stddev_by_level).view(len(self.output_variables), 1, 1)
        self.output_locations = torch.zeros_like(self.output_scales)
        self.forcing_scales = self._get_tensor_from_dataset(['day_progress_cos', 'day_progress_sin', 'toa_incident_solar_radiation', 'year_progress_cos', 'year_progress_sin'], [], stddev_by_level).view(5, 1, 1)
        self.forcing_locations = self._get_tensor_from_dataset(['day_progress_cos', 'day_progress_sin', 'toa_incident_solar_radiation', 'year_progress_cos', 'year_progress_sin'], [], mean_by_level).view(5, 1, 1)

    def _get_tensor_from_dataset(self, order: list, 
                                 aux_dataset_items: list, 
                                 dataset: xarray.Dataset, 
                                 aux_dataset: Optional[xarray.Dataset]=None
                                 ) -> torch.Tensor:
        tensor = []
        for item in order:
            if isinstance(item, str):
                if item in aux_dataset_items:
                    if aux_dataset is None:
                        raise ValueError("Need an Auxillary dataset to draw normalisation values from")
                    tensor.append(aux_dataset[item].item())
                else:
                    tensor.append(dataset[item].item())
            elif len(item) == 2:
                name, _ = item
                if name in aux_dataset_items:
                    if aux_dataset is None:
                        raise ValueError("Need an Auxillary dataset to draw normalisation values from")
                    tensor.append(aux_dataset[name].item())
                else:
                    tensor.append(dataset[name].item())
            elif len(item) == 3:
                name, _, level = item
                if name in aux_dataset_items:
                    if aux_dataset is None:
                        raise ValueError("Need an Auxillary dataset to draw normalisation values from")
                    tensor.append(aux_dataset[name].sel(level=int(level[1])).item())
                else:
                    tensor.append(dataset[name].sel(level=int(level[1])).item())
        return torch.tensor(tensor, device=self.model.device)

    def _outputs_from_input_tensor(self, inputs: torch.Tensor, index_array) -> torch.Tensor:
        data = []
        count = 0
        for idx in range(len(index_array)):
            count+=1
            # If the output is present in the input, the input needs to be extracted
            if index_array[idx] is not None:
                data.append(inputs[..., index_array[idx], :, :])
            # If the output is absent from the input, the global mean needs to be extracted
            else:
                item = self.output_variables[idx]
                data.append(torch.ones_like(inputs[..., 0, :, :])*(self._get_tensor_from_dataset([item], [], self._locations).item()))
        x = torch.stack(data, dim=-3).to(self.model.device)
        return x.squeeze()

    def _normalize(self,
                  data: torch.Tensor,
                  scales: torch.Tensor,
                  locations: Optional[torch.Tensor],
                  ) -> torch.Tensor:
        """Normalize variables using the given scales and (optionally) locations."""
        if locations is None:
            locations = torch.zeros_like(scales, device=self.model.device)
        return (data - locations)/scales
    
    def _unnormalize(self,
                  data: torch.Tensor,
                  scales: torch.Tensor,
                  locations: Optional[torch.Tensor],
                  ) -> torch.Tensor:
        """Normalize variables using the given scales and (optionally) locations."""
        if locations is None:
            locations = torch.zeros_like(scales, device=self.model.device)
        return (data * scales) + locations
    
    def _unnormalize_prediction_and_add_input(self, inputs: torch.Tensor,
                                              norm_prediction: torch.Tensor
                                              ) -> torch.Tensor:
        """Unnormalizes predictions and adds back the input (for residuals)."""
        # Inputs may not contain all the outputs. For those that it doesn't contain, 
        # multiply by the global stddev and add the mean back. Else, multiply by the 
        # stddev of the differences add the input back. Output scales contains the
        # sttdev of differences where applicable and global sttdev otherwise, so that
        # doesn't need any change. The means need to be added, which is done using 
        # _outputs_from_input_tensor
        residual = self._outputs_from_input_tensor(inputs, self.outputs_from_input_order)
        return self._unnormalize(norm_prediction, self.output_scales, residual)
    
    def _subtract_input_and_normalize_target(self, inputs: torch.Tensor,
                                             target: torch.Tensor
                                             ) -> torch.Tensor:
        """Subtracts input from target (for residuals) and normalizes."""
        # get correct output order, replacing invalid inputs with global mean
        # inputs is expected to have original order
        last_input = self._outputs_from_input_tensor(inputs, self.outputs_from_input_order)
        
        target_residual = self._normalize(target, self.output_scales, last_input)
        return target_residual

    def forward(self, inputs: torch.Tensor, forcings: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
        """Forward pass of the wrapped model."""

        # Normalise inputs and forcings
        inputs = inputs[..., self.input_permutation, :, :]
        norm_inputs = self._normalize(inputs, self.input_scales, self.input_locations)
        norm_forcings = self._normalize(forcings, self.forcing_scales, self.forcing_locations)

        # Concatenate inputs, forcings and node features along the channel dimension.
        combined_inputs = torch.cat((norm_inputs, norm_forcings, node_features), dim=-3).to(self.model.device).unsqueeze(0)
        norm_predictions = self.model(combined_inputs)
        # Unnormalize and add input (if residual).
        
        # norm_inputs has shape 176 in original order
        predictions = self._unnormalize_prediction_and_add_input(inputs, norm_predictions)

        return predictions

    def loss(self, inputs: torch.Tensor, outputs: torch.Tensor, targets: torch.Tensor, criterion: torch.nn.Module, **kwargs) -> torch.Tensor:
        """Computes the loss on normalized inputs and outputs.

        ----------
        Parameters
        ----------
        inputs : torch.Tensor
            Model inputs at time = 0. Precipitation values present for Operational Model
        outputs : torch.Tensor
            Model prediction obtained from forward()
        targets : torch.Tensor
            Model inputs at time = 1. Precipitation values present for Operational Model            
        criterion : torch.nn.Module
            Initialised loss function
        """

        # Normalise inputs and forcings
        # norm_inputs = self._normalize(inputs[..., self.input_permutation, :, :], self.input_scales, self.input_locations)
        # norm_forcings = self._normalize(forcings, self.forcing_scales, self.forcing_locations)
        
        # Concatenate inputs, forcings and node features along the channel dimension.
        # combined_inputs = torch.cat((inputs[..., self.input_permutation, :, :], forcings, node_features), dim=-3).to(self.model.device).unsqueeze(0)
        # combined_inputs_extracted_outputs = self._outputs_from_input_tensor(combined_inputs)
        
        # Targets has size 178, extracted_outputs should have size 83
        targets_extracted_outputs = self._outputs_from_input_tensor(targets, self.outputs_from_full_input_order)
        
        inputs = inputs[..., self.input_permutation, :, :] # Inputs now has size 176 and is in original order
        
        norm_actual_residuals = self._subtract_input_and_normalize_target(inputs, targets_extracted_outputs)
        norm_predicted_residuals = self._subtract_input_and_normalize_target(inputs, outputs)
                
        loss = criterion(norm_predicted_residuals, norm_actual_residuals, **kwargs)
        return loss
