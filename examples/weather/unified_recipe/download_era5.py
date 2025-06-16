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
import dask
import fsspec
import hydra
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from omegaconf import DictConfig, OmegaConf
import data_utils
import os

from utils import get_filesystem

# Add eval to OmegaConf TODO: Remove when OmegaConf is updated
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base="1.2", config_path="conf", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    # Resolve config so that all values are concrete
    OmegaConf.resolve(cfg)

    # Make single and pressure level variables list if not already
    if not cfg.dataset.single_level_variables:
        cfg.dataset.single_level_variables = []
    if not cfg.dataset.pressure_level_variables:
        cfg.dataset.pressure_level_variables = []

    # Get fsspec filesystem
    save_fs = get_filesystem(
        cfg.filesystem.type,
        cfg.filesystem.key,
        cfg.filesystem.endpoint_url,
        cfg.filesystem.region_name,
    )

    # Get file system mapper
    save_mapper = save_fs.get_mapper(cfg.dataset.dataset_filename)
    print(save_mapper)
    # Get ARCO ERA5 dataset
    arco_filename = (
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
    )
    # gs_fs = fsspec.filesystem("gs")
    # arco_era5 = xr.open_zarr(gs_fs.get_mapper(arco_filename), consolidated=True)
    os.environ['NO_GCE_CHECK'] = 'true'
    arco_era5 = xr.open_zarr(
        arco_filename, 
        storage_options={'anon': True, 'token': 'anon'}, 
        consolidated=True
    )
    # Drop variables that are not needed
    for variable in arco_era5.variables:
        if (
            variable
            not in cfg.dataset.single_level_variables
            + cfg.dataset.pressure_level_variables
        ):
            arco_era5 = arco_era5.drop(variable)

    # Make encoding for chunking pressure level variables
    encoding = {}
    for variable in cfg.dataset.pressure_level_variables:
        encoding[variable] = {"chunks": (1, 1, 721, 1440)}

    # Subsample time
    arco_era5 = arco_era5.sel(
        time=slice(
            datetime.datetime.strptime(cfg.dataset.years[0], "%Y-%m-%d"),
            datetime.datetime.strptime(cfg.dataset.years[1], "%Y-%m-%d"),
        ), 
        level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    )
    arco_era5 = arco_era5.sel(
        time=arco_era5.time.dt.hour.isin(np.arange(0, 24, cfg.dataset.dt))
    )
    lon_grid, lat_grid = np.meshgrid(arco_era5.longitude, arco_era5.latitude)

    # Convert grids to radians
    lon_rad_grid = np.deg2rad(lon_grid)
    lat_rad_grid = np.deg2rad(lat_grid)

    # Calculate spherical coordinates
    cos_lat_grid = np.cos(lat_rad_grid)
    sin_lon_grid = np.sin(lon_rad_grid)
    cos_lon_grid = np.cos(lon_rad_grid)

    # Expand grids to include the time dimension
    time_len = len(arco_era5.time)
    cos_lat_grid_time = np.tile(cos_lat_grid, (time_len, 1, 1))
    sin_lon_grid_time = np.tile(sin_lon_grid, (time_len, 1, 1))
    cos_lon_grid_time = np.tile(cos_lon_grid, (time_len, 1, 1))
    
    # Add the grids as data variables
    arco_era5 = arco_era5.assign(
        {
            "cos_latitude": (("time", "latitude", "longitude"), cos_lat_grid_time),
            "cos_longitude": (("time", "latitude", "longitude"), cos_lon_grid_time),
            "sin_longitude": (("time", "latitude", "longitude"), sin_lon_grid_time),        
        }
    )
    arco_era5 = arco_era5.expand_dims({'batch':1})
    date_time = arco_era5.time.values

    date_time_2d = np.array([date_time])
    arco_era5 = arco_era5.assign_coords({'datetime': (('batch', 'time'), date_time_2d)})
    
    data_utils.add_derived_vars(arco_era5)
    arco_era5 = arco_era5.drop_vars(["datetime", 'year_progress', 'day_progress'])
    arco_era5 = arco_era5.squeeze('batch', drop=True)
    # Save data
    print("Saving Data")
    with ProgressBar():
        # TODO: Remove single_threaded when machine updated
        if cfg.dataset.single_threaded:
            with dask.config.set(scheduler="single-threaded"):
                arco_era5.to_zarr(save_mapper, consolidated=True, encoding=encoding)
        else:
            arco_era5.to_zarr(save_mapper, consolidated=True, encoding=encoding)


if __name__ == "__main__":
    main()
