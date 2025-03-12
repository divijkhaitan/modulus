from typing import Iterable, Dict, List, Optional, Tuple
import torch
import numpy as np



def normalized_level_weights(data: Iterable) -> torch.Tensor:
  """Weights proportional to pressure at each level."""
  levels = torch.tensor(data)
  return levels / levels.mean()

def get_latitude_weights(latitude):
  sorted_order = sorted(latitude)
  diff = sorted_order[1] - sorted_order[0]
  if (sorted_order != np.linspace(sorted_order[0], sorted_order[-1], len(latitude))).any(): 
    raise ValueError(f'Latitude vector {latitude} does not start/end at +- 90 degrees.')
  weights = [np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(diff/2)) if lat not in [-90, 90] 
             else np.sin(np.deg2rad(diff/4)) ** 2 for lat in latitude]
  return np.array(weights)


def get_weights(shape: Tuple[int, int, int], lat_coords: List, level_mapping: List, 
                  variable_weights: Optional[Dict] = None):
    """
    Get weights for a tensor based on coordinates and channel characteristics.
    
    Args:
        lat_coords (array-like): Latitude coordinates corresponding to the lat dimension
        level_mapping (list): For each channel, specifies which atmospheric level it represents
                             (None if the variable doesn't have a level)
        variable_weights (dict): Dictionary mapping variable names to their weights
                                
    Returns:
        torch.Tensor: Weighted tensor with same shape as input
    """
    channels, lat_len, lon_len = shape
    
    # Create empty tensor to hold weights
    weights = torch.ones((channels, lat_len, lon_len))
    
    # Latitude weights (same for all channels)
    lat_weights = get_latitude_weights(lat_coords)
    
    # Broadcast latitude weights to match tensor dimensions (channels, lat, lon)
    lat_weights_tensor = torch.tensor(lat_weights, dtype=torch.float32)
    lat_weights_tensor = lat_weights_tensor.view(1, lat_len, 1).expand(channels, lat_len, lon_len)
    
    # Apply latitude weights
    weights *= lat_weights_tensor
    partition_value = np.mean(np.unique([level for level in level_mapping if level is not None]))
    level_mapping = [level/partition_value if level is not None else 1 for level in level_mapping]
    # Level weights for channels that have levels
    for c in range(channels):
        weights[c, :, :] *= level_mapping[c]
    
    # Variable-specific weights
    for c, var_name in enumerate(variable_weights.keys()):
        weights[c, :, :] *= variable_weights[var_name]
    
    return weights

class WeightedMSELoss(torch.nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.weights = weights.squeeze()
        # self.criterion = torch.nn.MSELoss()

    def forward(self, predictions, targets):
        assert(targets.shape == predictions.shape)
        assert(self.weights.shape == targets.shape[:len(self.weights.shape)])
        return torch.mean(torch.pow((predictions - targets), 2)*self.weights)