import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gc 

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Optional
try:
    import cartopy.crs as ccrs
    CARTOPY_INSTALLED = True
except ImportError:
    CARTOPY_INSTALLED = False

def plot_error_timeseries(
    groundtruth_path: str,
    forecast_paths: Dict[str, str],
    output_folder: str,
    sub_region: Optional[Dict[str, slice]] = None,
    suffix: Optional[str] = None
) -> None:
    """
    Calculates forecast error vs. lead time and plots it as a time series.

    This function computes the Mean Absolute Error (MAE) for each model at
    various lead times and generates a single line plot comparing all models.

    Args:
        groundtruth_path (str): Path to the ground truth zarr dataset.
        forecast_paths (Dict[str, str]): Dictionary of model names and paths.
        output_folder (str): Folder to save the output plot.
        sub_region (Optional[Dict[str, slice]]): Optional dictionary to select
            a spatial sub-region for analysis.
            Example: {'latitude': slice(40, 5), 'longitude': slice(60, 100)}
        suffix (Optional[str]): Optional string to append to the output filename.
    """
    # --- 1. Setup and Data Loading ---
    os.makedirs(output_folder, exist_ok=True)
    try:
        gt_ds = xr.open_zarr(groundtruth_path)
        # Ensure coordinates are sorted for robust processing
        gt_ds.coords['step'] = gt_ds.coords['step'] + pd.to_timedelta(6, unit='h')
        gt_ds = gt_ds.sortby(['latitude', 'longitude'])
        if sub_region:
            print(f"Subsetting ground truth to region: {sub_region}")
            gt_ds = gt_ds.sel(sub_region)
    except Exception as e:
        print(f"FATAL: Could not load or subset ground truth data. {e}")
        return

    # --- 2. Initialize Data Structures ---
    lead_times_h = np.arange(6, 73, 6)
    lead_times_td = [pd.to_timedelta(f'{h}h') for h in lead_times_h]
    model_names = ['Persistence (24h)'] + list(forecast_paths.keys())
    # DataFrame to store the scalar error values
    errors_df = pd.DataFrame(index=model_names, columns=lead_times_h, dtype=float)

    # --- 3. Calculate Errors for each Model and Lead Time ---
    print("Calculating errors...")
    
    # --- Persistence Model ---
    try:
        base_time_diff = np.median(np.diff(gt_ds['base_time'].values))
        shift_steps = int(round(pd.to_timedelta('24h') / base_time_diff))
        persistence_fc = gt_ds.shift(base_time=-shift_steps)
        aligned_gt_p, aligned_pers = xr.align(gt_ds, persistence_fc, join='inner', copy=False)
        for lead_h, lead_td in zip(lead_times_h, lead_times_td):
            gt_at_lead = aligned_gt_p['2m_temperature'].sel(step=lead_td, method='nearest')
            pers_at_lead = aligned_pers['2m_temperature'].sel(step=lead_td, method='nearest')
            # Calculate the single MAE value over the spatial and time dimensions
            error = float(xr.ufuncs.fabs(gt_at_lead - pers_at_lead).mean().compute())
            errors_df.loc['Persistence (24h)', lead_h] = error
        print("  - Calculated Persistence error.")
    except Exception as e:
        print(f"Warning: Could not compute persistence error. {e}")
    print(errors_df)
    exit()
    # --- Forecast Models ---
    for model_name, forecast_path in forecast_paths.items():
        try:
            if forecast_path.endswith('.zarr'):
                fc_ds = xr.open_zarr(forecast_path).sortby(['latitude', 'longitude'])
                fc_ds.coords['step'] = fc_ds.coords['step'] + pd.to_timedelta(6, unit='h')
            else:
                fc_ds = xr.open_dataset(forecast_path).sortby(['latitude', 'longitude'])
                if "t2m" in fc_ds.data_vars: fc_ds = fc_ds.rename({'t2m': '2m_temperature'})
                if "time" in fc_ds.dims: fc_ds = fc_ds.rename({'time': 'base_time'})
                coords_to_drop = [c for c in ['number', 'heightAboveGround', 'surface', 'valid_time'] if c in fc_ds.coords]
                if coords_to_drop: fc_ds = fc_ds.drop_vars(coords_to_drop)
                
            if sub_region:
                fc_ds = fc_ds.sel(sub_region)

            regridded_fc = fc_ds.interp_like(gt_ds, method="linear")
            aligned_fc, aligned_gt = xr.align(regridded_fc, gt_ds, join='inner', copy=False)

            for lead_h, lead_td in zip(lead_times_h, lead_times_td):
                gt_at_lead = aligned_gt['2m_temperature'].sel(step=lead_td, method='nearest')
                fc_at_lead = aligned_fc['2m_temperature'].sel(step=lead_td, method='nearest')
                error = float(xr.ufuncs.fabs(gt_at_lead - fc_at_lead).mean().compute())
                errors_df.loc[model_name, lead_h] = error
            print(f"  - Calculated {model_name} error.")
        except Exception as e:
            print(f"Warning: Could not process model '{model_name}'. {e}")

    # --- 4. Plotting the Time Series ---
    print("\nGenerating time series plot...")
    
    # Transpose the DataFrame so lead times are on the x-axis
    errors_to_plot = errors_df.T
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use seaborn for a slightly nicer look, with markers
    sns.lineplot(data=errors_to_plot, ax=ax, marker='o', markersize=8)
    
    # Formatting the plot
    region_str = "Global" if not sub_region else "Sub-Region"
    title_suffix = f" ({suffix})" if suffix else ""
    ax.set_title(f'Forecast Error (MAE) vs. Lead Time for 2m Temperature ({region_str}){title_suffix}', fontsize=16)
    ax.set_xlabel('Lead Time (hours)', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (K)', fontsize=12)
    ax.legend(title='Forecast Model')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Ensure x-axis ticks match the lead times
    ax.set_xticks(lead_times_h)
    
    plt.tight_layout()
    
    # Save the figure
    filename_suffix = f"_{suffix}" if suffix else ""
    output_filename = os.path.join(output_folder, f"error_timeseries{filename_suffix}.png")
    plt.savefig(output_filename, dpi=150)
    print(f"Successfully saved plot to: {output_filename}")
    
    # Display the final error data
    print("\n--- Mean Absolute Error (K) ---")
    print(errors_df.to_string(float_format="%.4f"))

if __name__ == "__main__":
    output_directory = 'total_rmse_plots'
    groundtruth_path = '/Datastorage/divij.khaitan_asp25/model_package_India-Finetune-100epoch/groundtruth.zarr'
    forecast_paths = {
        "gc-finetuned": '/Datastorage/divij.khaitan_asp25/model_package_India-Finetune-100epoch/predicted.zarr',
        "gc-base": '/Datastorage/divij.khaitan_asp25/model_package_India-Finetune-100epoch/predicted_base.zarr',
        "ncep": '/Datastorage/divij.khaitan_asp25/forecasts_2022/ncep_forecasts.nc',
        "imd": '/Datastorage/divij.khaitan_asp25/forecasts_2022/imd_forecasts.nc',
        "hres": '/Datastorage/divij.khaitan_asp25/forecasts_2022/hres_forecasts.nc' # Add hres if needed
    }

    india_sub_region = {'latitude': slice(7.5, 37.5), 'longitude': slice(67.5, 97.5)}

    print(f"Cartopy installed: {CARTOPY_INSTALLED}")
    if not CARTOPY_INSTALLED:
        print("NOTE: Cartopy not found. Maps will be plotted without coastlines. For better plots, run 'pip install cartopy'")


    plot_error_timeseries(
        groundtruth_path=groundtruth_path,
        forecast_paths=forecast_paths,
        output_folder=output_directory,
        sub_region=india_sub_region,
        suffix='India',
    )