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

import datetime
from pathlib import Path
from typing import List, Tuple, Union, Dict

import dask
import fsspec
import hydra
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from omegaconf import DictConfig, OmegaConf

# Add eval to OmegaConf TODO: Remove when OmegaConf is updated
OmegaConf.register_new_resolver("eval", eval)

from utils import get_filesystem  # Assuming you have this utility


class CurateERA5:
    """
    Curate a Zarr ERA5 dataset to a Zarr dataset used for training global weather
    models.
    """

    def __init__(
        self,
        variable_groups: Dict[str, List[Union[str, List[Union[str, int]]]]],
        dataset_filename: Union[str, Path] = "./data.zarr",
        fs: fsspec.filesystem = fsspec.filesystem("file"),
        curated_dataset_filename: Union[str, Path] = "./curated_data.zarr",
        curated_fs: fsspec.filesystem = fsspec.filesystem("file"),
        date_range: Tuple[str, str] = ("2000-01-01", "2001-01-01"),
        dt: int = 1,  # 1 hour
        chunk_channels_together: bool = True,
        single_threaded: bool = False,
        resolution: float = 0.25
    ):
        super().__init__()

        # Store parameters
        self.variable_groups = variable_groups
        self.dataset_filename = dataset_filename
        self.fs = fs
        self.curated_dataset_filename = curated_dataset_filename
        self.curated_fs = curated_fs
        self.date_range = date_range
        assert dt in [1, 3, 6, 12], "dt must be 1, 3, 6, or 12"
        self.dt = dt
        self.chunk_channels_together = chunk_channels_together
        self.single_threaded = single_threaded

        # Open dataset to do curation from
        mapper = fs.get_mapper(self.dataset_filename)
        self.era5 = xr.open_zarr(mapper, consolidated=True)
        old_lats = self.era5['latitude'].values
        old_lons = self.era5['longitude'].values
        new_lats = np.arange(old_lats.min(), old_lats.max() + 1e-8, resolution)
        new_lons = np.arange(old_lons.min(), old_lons.max() + 1e-8, resolution)
        self.era5 = self.era5.interp({'latitude': new_lats, 'longitude': new_lons}, 
                                 method='linear',
                                 kwargs={'fill_value': None})
        # Subset variables (this speeds up chunking)
        needed_variables = ["latitude", "longitude", "time", "level"]

        # Helper function to extract variables
        def extend_needed_variables(groups_dict):
            for group_vars in groups_dict.values():
                for variable in group_vars:
                    if isinstance(variable, list):  # Handle pressure level variables
                        needed_variables.append(variable[0])
                    else:
                        needed_variables.append(variable)

        extend_needed_variables(self.variable_groups)

        for variable in self.era5.variables:
            if variable not in needed_variables:
                self.era5 = self.era5.drop_vars(variable)

        # Chunk data
        self.era5 = self.era5.sel(
            time=slice(
                datetime.datetime.strptime(date_range[0], "%Y-%m-%d"),
                datetime.datetime.strptime(date_range[1], "%Y-%m-%d"),
            )
        )
        self.era5 = self.era5.sel(
            time=self.era5.time.dt.hour.isin(np.arange(0, 24, self.dt))
        )
        self.era5 = self.era5.chunk(
            {"time": 1, "level": 1, "latitude": 721, "longitude": 1440}
        )

    def _process_variable_group(self, variables: List[Union[str, List[Union[str, int]]]], group_name: str):
        """Processes and concatenates variables for a given group."""
        xarray_variables = []
        for variable in sorted(variables):
            if isinstance(variable, list):  # Handle pressure level variables
                var_name = variable[0]
                levels = variable[1]
                if isinstance(levels, int): # if only one pressure level
                    levels = [levels]
                for level in sorted(levels):  # Iterate through pressure levels
                    pressure_variable = self.era5[var_name].sel(level=level)
                    pressure_variable = pressure_variable.expand_dims(f"{group_name}_channel", axis=1) # Add channel dim
                    # Add a coordinate for the channel, using the variable name and level
                    pressure_variable = pressure_variable.assign_coords({f"{group_name}_channel": [f"{var_name}_{level}"]})
                    xarray_variables.append(pressure_variable)

            else:  # Handle single-level variables
                single_variable = self.era5[variable]
                single_variable = single_variable.expand_dims(f"{group_name}_channel", axis=1)
                # Add a coordinate for the channel, using the variable name
                single_variable = single_variable.assign_coords({f"{group_name}_channel": [variable]})
                xarray_variables.append(single_variable)
        return xr.concat(xarray_variables, dim=f"{group_name}_channel", coords='minimal')



    def __call__(self):
        """
        Generate the zarr array
        """

        # Check if already exists
        if self.fs.exists(self.curated_dataset_filename):
            print(f"Zarr file {self.curated_dataset_filename} already exists")
            return

        # Create the curated dataset
        self.era5_subset = xr.Dataset()

        # Define the expected groups
        expected_groups = ["constants", "inputs_surface", "inputs_pressure_levels", "forcings", "node_features"]

        # Find total_precipitation index
        tp_channel_index = -1
        if "inputs_surface" in self.variable_groups:
            # Need to adjust to account for multiple pressure levels in the same group
            tp_index_in_list = -1
            try:
                tp_index_in_list = self.variable_groups["inputs_surface"].index("total_precipitation")
            except ValueError:
                pass

            if tp_index_in_list >=0:
                # tp_channel_index = 0  # No, this is not necessarily 0 now
                count = 0
                for i in range(tp_index_in_list):  # Iterate up to "tp"
                    var = self.variable_groups["inputs_surface"][i]
                    if isinstance(var, list):
                        if isinstance(var[1], int):
                            count += 1
                        else:
                            count += len(var[1])  # Add the number of levels
                    else:
                        count += 1
                # tp_channel_index = count  # Corrected index, before the TP entry
                # Now count constants correctly as well
                constants_count = len(self.variable_groups['constants'])
                tp_channel_index = constants_count + count # combined


        # Process each group and add to the dataset
        for group_name in expected_groups:
            if group_name in self.variable_groups:
                variables = self.variable_groups[group_name]
                self.era5_subset[group_name] = self._process_variable_group(variables, group_name)

                # Chunk channels
                if self.chunk_channels_together:
                    channel_chunk_size = self.era5_subset[group_name].sizes[
                        f"{group_name}_channel"
                    ]
                else:
                    channel_chunk_size = 1

                self.era5_subset[group_name] = self.era5_subset[group_name].chunk(
                    {f"{group_name}_channel": channel_chunk_size}
                )
            else: # if group does not exist
                self.era5_subset[group_name] = xr.DataArray(
                    [],
                    coords={
                        "time": self.era5.time,
                        f"{group_name}_channel": [],
                        "latitude": self.era5.latitude,
                        "longitude": self.era5.longitude,
                    },
                    dims=["time", f"{group_name}_channel", "latitude", "longitude"],
                    name=group_name
                ).chunk({"time": 1, f"{group_name}_channel": 1})

        self.era5_subset["time"] = self.era5["time"]

        # Add attributes for total precipitation index
        self.era5_subset.attrs["tp_channel_index"] = tp_channel_index

        # Save
        mapper = self.curated_fs.get_mapper(self.curated_dataset_filename)
        delayed_obj = self.era5_subset.to_zarr(mapper, consolidated=True, compute=False)

        # Wait for save to finish (Single-threaded legacy issue)
        with ProgressBar():
            if self.single_threaded:
                with dask.config.set(scheduler="single-threaded"):
                    delayed_obj.compute()
            else:
                delayed_obj.compute()


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Resolve config so that all values are concrete
    OmegaConf.resolve(cfg)

    # Get filesystem
    fs = get_filesystem(
        cfg.filesystem.type,
        cfg.filesystem.key,
        cfg.filesystem.endpoint_url,
        cfg.filesystem.region_name,
    )

    # Make train data
    curate_train_era5 = CurateERA5(
        variable_groups=cfg.curated_dataset.variable_groups,
        dataset_filename=cfg.dataset.dataset_filename,
        fs=fs,
        curated_dataset_filename=cfg.curated_dataset.train_dataset_filename,
        curated_fs=fs,
        date_range=cfg.curated_dataset.train_years,
        dt=cfg.curated_dataset.dt,
        chunk_channels_together=cfg.curated_dataset.chunk_channels_together,
        resolution=cfg.resolution
    )
    curate_train_era5()

    # Make validation data
    curate_val_era5 = CurateERA5(
        variable_groups=cfg.curated_dataset.variable_groups,
        dataset_filename=cfg.dataset.dataset_filename,
        fs=fs,
        curated_dataset_filename=cfg.curated_dataset.val_dataset_filename,
        curated_fs=fs,
        date_range=cfg.curated_dataset.val_years,
        dt=cfg.curated_dataset.dt,
        chunk_channels_together=cfg.curated_dataset.chunk_channels_together,
        resolution=cfg.resolution
    )
    curate_val_era5()


if __name__ == "__main__":
    main()