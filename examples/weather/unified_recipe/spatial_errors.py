import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Optional
plt.rcParams['font.size'] = '20'

def filter_dates(ds, start_date="2022-01-01", end_date="2023-01-01", midnight=False):
    """
    Filters the dataset by date. Assumes 'base_time' is a dimension
    and coordinate that can be indexed.
    """
    time_slice = slice(start_date, end_date)
    
    if midnight:
        ds_filtered = ds.sel(base_time=time_slice)
        base_times = pd.to_datetime(ds_filtered['base_time'].values)
        mask = (base_times.time == pd.to_datetime("00:00").time()) | \
               (base_times.time == pd.to_datetime("12:00").time())
        return ds_filtered.sel(base_time=base_times[mask])
    else:
        return ds.sel(base_time=time_slice)

try:
    import cartopy.crs as ccrs
    CARTOPY_INSTALLED = True
except ImportError:
    CARTOPY_INSTALLED = False

def plot_spatial_forecast_errors(
    groundtruth_path: str,
    forecast_paths: Dict[str, str],
    output_folder: str,
    sub_region: Optional[Dict[str, slice]] = None,
    suffix: Optional[str] = None
) -> None:
    """
    Generates spatial error map plots for each lead time.

    For each lead time, a single plot file is created containing subplots
    for each forecast model's 2D spatial error map.

    Args:
        groundtruth_path (str): Path to the ground truth zarr dataset.
        forecast_paths (Dict[str, str]): Dictionary of model names and paths.
        output_folder (str): Folder to save the output plots.
        sub_region (Optional[Dict[str, slice]]): Optional dictionary to select
            a spatial sub-region for analysis.
            Example: {'latitude': slice(40, 5), 'longitude': slice(60, 100)}
        suffix (Optional[str]): Optional string to append to output filenames.
    """
    os.makedirs(output_folder, exist_ok=True)

    try:
        gt_ds = xr.open_zarr(groundtruth_path).sortby(['latitude', 'longitude'])
        gt_ds.coords['step'] = gt_ds.coords['step'] + pd.to_timedelta(6, unit='h')
        if sub_region:
            print(f"Subsetting ground truth to region: {sub_region}")
            gt_ds = gt_ds.sel(sub_region)
    except Exception as e:
        print(f"FATAL: Could not load or subset ground truth data. {e}")
        return
    gt_ds = filter_dates(gt_ds, midnight=True)
    lead_times_h = np.arange(6, 73, 6)
    lead_times_td = [pd.to_timedelta(f'{h}h') for h in lead_times_h]
    common_lats = np.linspace(7.5, 37.5, 121)
    common_lons = np.linspace(67.5, 97.5, 121)
    model_names = ['persistence'] + list(forecast_paths.keys())
    temporal_errors_data = []

    for lead_h, lead_td in zip(lead_times_h, lead_times_td):
        print(f"\n--- Processing Lead Time: {lead_h} hours ---")
        error_maps = {}

        try:
            base_time_diff = np.median(np.diff(gt_ds['base_time'].values))
            shift_steps = int(round(pd.to_timedelta('24h') / base_time_diff))
            
            persistence_fc = gt_ds.shift(base_time=-shift_steps)
            aligned_gt_p, aligned_pers = xr.align(gt_ds, persistence_fc, join='inner', copy=False)

            gt_at_lead_p = aligned_gt_p['2m_temperature'].sel(step=lead_td, method='nearest')
            pers_at_lead = aligned_pers['2m_temperature'].sel(step=lead_td, method='nearest')
            
            persistence_error_map = xr.ufuncs.fabs(gt_at_lead_p - pers_at_lead).mean(dim='base_time')
            error_maps['persistence'] = persistence_error_map.compute()
            persistence_temporal_error = xr.ufuncs.fabs(gt_at_lead_p - pers_at_lead).mean(dim=['base_time', 'latitude', 'longitude'])
            temporal_errors_data.append({
                'model': 'persistence',
                'lead_time_hours': lead_h,
                'temporal_error': float(persistence_temporal_error.values)
            })
        except Exception as e:
            print(f"Warning: Could not compute persistence error map. {e}")
        # print("Groundtruth")
        # print(gt_ds)
        
        # --- Calculate Error Map for each Forecast Model ---
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
                fc_ds = filter_dates(fc_ds, midnight=True)
                if sub_region:
                    fc_ds = fc_ds.sel(sub_region)
                # print(model_name)
                # print(fc_ds)
                regridded_fc = fc_ds.interp(latitude=common_lats, longitude=common_lons, method="linear")
                # print(regridded_fc)
                aligned_fc, aligned_gt = xr.align(regridded_fc, gt_ds, join='inner', copy=False)
                # print(aligned_fc, aligned_gt)
                gt_at_lead = aligned_gt['2m_temperature'].sel(step=lead_td, method='nearest')
                fc_at_lead = aligned_fc['2m_temperature'].sel(step=lead_td, method='nearest')
                
                # Take the mean over the time dimension to get a single 2D error map
                model_error_map = xr.ufuncs.fabs(gt_at_lead - fc_at_lead).mean(dim='base_time')
                error_maps[model_name] = model_error_map.compute()
                temporal_error = xr.ufuncs.fabs(gt_at_lead - fc_at_lead).mean(dim=['base_time', 'latitude', 'longitude'])
                temporal_errors_data.append({
                    'model': model_name,
                    'lead_time_hours': lead_h,
                    'temporal_error': float(temporal_error.values)
                })
            except Exception as e:
                print(f"Warning: Could not process model '{model_name}'. {e}")
            
        if not error_maps:
            print(f"No data to plot for lead time {lead_h}h. Skipping.")
            continue

        # # --- 4. Plotting ---
        # num_models = len(error_maps)
        # ncols = 2
        # nrows = int(np.ceil(num_models / ncols))
        
        # # Determine shared color scale for all subplots in this figure
        # vmin = min(m.min() for m in error_maps.values())
        # vmax = max(m.max() for m in error_maps.values())

        # proj = ccrs.PlateCarree() if CARTOPY_INSTALLED else None
        # fig, axes = plt.subplots(
        #     nrows=nrows, ncols=ncols, figsize=(ncols * 12, nrows * 12),
        #     subplot_kw={'projection': proj}, squeeze=False
        # )
        # axes = axes.flatten()

        # for i, (model_name, error_map) in enumerate(error_maps.items()):
        #     ax = axes[i]
        #     # Use xarray's powerful plotting, which understands coordinates.
        #     plot_kwargs = {
        #         'ax': ax,
        #         'vmin': vmin, 'vmax': vmax,
        #         'cmap': 'viridis',
        #         'add_colorbar': False # We will add a shared colorbar later
        #     }
        #     if CARTOPY_INSTALLED:
        #         plot_kwargs['transform'] = ccrs.PlateCarree()

        #     error_map.plot.pcolormesh(**plot_kwargs)
        #     ax.set_title(f'Model: {model_name}\nMAE: {error_map.mean().item():.3f} K')
            
        #     if CARTOPY_INSTALLED:
        #         ax.coastlines()
        #         ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)

        # # Clean up unused subplots
        # for i in range(num_models, len(axes)):
        #     fig.delaxes(axes[i])

        # # Add a single, shared colorbar
        # fig.subplots_adjust(right=0.9)
        # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        # sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin, vmax))
        # cbar = fig.colorbar(sm, cax=cbar_ax)
        # cbar.set_label('Mean Absolute Error (K)', fontsize=12)

        # # Add a main title for the entire figure
        # region_str = "Global" if not sub_region else "Sub-Region"
        # fig.suptitle(f'Spatial Forecast Error for Lead Time: {lead_h} Hours ({region_str})', fontsize=16)
        
        # # Save the figure
        # filename_suffix = f"_{suffix}" if suffix else ""
        # output_filename = os.path.join(output_folder, f"error_map_lead_{lead_h:02d}h{filename_suffix}.png")
        # plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        # print(f"Saved plot: {output_filename}")
        # plt.close(fig) # Close the figure to free memory
    temporal_errors_df = pd.DataFrame(temporal_errors_data)
    
    if not temporal_errors_df.empty:
        # --- Create Time Series Plot ---
        plt.figure(figsize=(12, 8))
        
        # Plot each model's temporal error vs lead time
        for model in temporal_errors_df['model'].unique():
            model_data = temporal_errors_df[temporal_errors_df['model'] == model]
            print(model_data['temporal_error'])
            plt.plot(model_data['lead_time_hours'], model_data['temporal_error'], 
                    marker='o', linewidth=2, markersize=6, label=model)
        
        plt.xlabel('Lead Time (hours)', fontsize=14)
        plt.ylabel('Mean Absolute Error (K)', fontsize=14)
        plt.title('Temporal Forecast Error vs Lead Time', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the time series plot
        filename_suffix = f"_{suffix}" if suffix else ""
        timeseries_filename = os.path.join(output_folder, f"temporal_error_timeseries{filename_suffix}.png")
        plt.savefig(timeseries_filename, dpi=150, bbox_inches='tight')
        print(f"Saved time series plot: {timeseries_filename}")
        plt.close()
        
        # --- Save DataFrame to CSV ---
        csv_filename = os.path.join(output_folder, f"temporal_errors{filename_suffix}.csv")
        temporal_errors_df.to_csv(csv_filename, index=False)
        print(f"Saved temporal errors DataFrame: {csv_filename}")
        
        # Print summary statistics
        print("\n--- Temporal Error Summary ---")
        print(temporal_errors_df.groupby('model')['temporal_error'].agg(['mean', 'std', 'min', 'max']).round(4))
    
    return temporal_errors_df

if __name__ == '__main__':
    
    output_directory = 'rmse_heatmaps'
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

    plot_spatial_forecast_errors(
        groundtruth_path=groundtruth_path,
        forecast_paths=forecast_paths,
        output_folder=output_directory,
        sub_region=india_sub_region,
        suffix="no_interp"
    )