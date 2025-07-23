import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gc 

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

def plot_and_save_spatial_rmse_heatmaps(
    ground_truth_path,
    forecast_paths,
    variable_name="2m_temperature",
    output_dir="rmse_heatmaps",
    suffix='',
    date_filter_kwargs=None
):
    os.makedirs(output_dir, exist_ok=True)
    if date_filter_kwargs is None: date_filter_kwargs = {}

    common_lats = np.linspace(7.5, 37.5, 61)
    common_lons = np.linspace(67.5, 97.5, 61)

    print("Lazily opening all datasets...")
    chunks = {'base_time': 10, 'time': 10} 

    with xr.open_zarr(ground_truth_path, chunks=chunks, consolidated=True) as ds:
        ds.coords['step'] = ds.coords['step'] + pd.to_timedelta(6, unit='h')
        gt_ds_filtered = filter_dates(ds, **date_filter_kwargs)

    forecast_dsets = {}
    for name, path in forecast_paths.items():
        if path.endswith('.zarr'):
            ds = xr.open_zarr(path, chunks=chunks, consolidated=True)
            ds.coords['step'] = ds.coords['step'] + pd.to_timedelta(6, unit='h')
        elif path.endswith('.nc'):
            ds = xr.open_dataset(path, chunks=chunks)
            if "t2m" in ds.data_vars: ds = ds.rename({'t2m': '2m_temperature', 'tp': 'total_precipitation'})
            if "time" in ds.dims: ds = ds.rename({'time': 'base_time'})
            coords_to_drop = [c for c in ['number', 'heightAboveGround', 'surface', 'valid_time'] if c in ds.coords]
            if coords_to_drop: ds = ds.drop_vars(coords_to_drop)
        print(ds)
        forecast_dsets[name] = filter_dates(ds, **date_filter_kwargs)
    exit()
    first_step_val = pd.to_timedelta(gt_ds_filtered.step.data[0])
    gt_valid_time_da = gt_ds_filtered[variable_name].sel(step=first_step_val)
    actual_valid_times = gt_valid_time_da.base_time + first_step_val
    processed_gt_da = gt_valid_time_da.assign_coords(time=actual_valid_times).swap_dims({'base_time': 'time'})

    all_steps_values = set()
    for ds in forecast_dsets.values():
        if 'step' in ds.coords:
            steps = pd.to_timedelta(ds.step.values)
            all_steps_values.update(s for s in steps if s != pd.Timedelta(0))
    
    if not all_steps_values:
        print("No forecast lead times found to plot. Exiting.")
        return

    for selected_forecast_step in sorted(list(all_steps_values)):
        hr_lead = int(selected_forecast_step / np.timedelta64(1, 'h'))
        print(f"\nProcessing lead time: {hr_lead} hours...")

        rmse_data = {}
        all_rmse_values = []
        
        # --- NEW: 24-Hour Persistence Calculation ---
        try:
            # Use a consistent reference model for the persistence timeline (e.g., the first one)
            ref_model_name = next(iter(forecast_paths))
            print(f"Calculating Persistence (24h) Benchmark using '{ref_model_name}' as reference...")
            ref_model_ds = forecast_dsets[ref_model_name]

            if 'step' in ref_model_ds.coords and selected_forecast_step in pd.to_timedelta(ref_model_ds.step.data):
                ref_base_times = ref_model_ds.sel(step=selected_forecast_step)['base_time'].data

                # Calculate valid times and the NEW persistence times (24h before valid time)
                valid_times = ref_base_times + selected_forecast_step
                persistence_times = valid_times - pd.Timedelta(days=1)

                # Interpolate ground truth to these two sets of times
                gt_at_valid_time = processed_gt_da.interp(time=valid_times, method='linear').compute()
                gt_persistence_forecast = processed_gt_da.interp(time=persistence_times, method='linear').compute()

                # Calculate RMSE
                # Ensure the persistence forecast data has the same time coordinate as the valid data for subtraction
                gt_persistence_forecast['time'] = gt_at_valid_time['time']
                error_persistence = gt_at_valid_time - gt_persistence_forecast
                rmse_persistence = np.sqrt((error_persistence**2).mean(dim='time', skipna=True))
                rmse_data['persistence'] = rmse_persistence
                all_rmse_values.extend(rmse_persistence.values.flatten())
                print("  - Persistence RMSE calculated.")

                del gt_at_valid_time, gt_persistence_forecast, error_persistence
                gc.collect()
        except Exception as e:
            print(f"  - Could not calculate persistence benchmark: {e}")

        # --- Model RMSE Calculation for this step ---
        for model_name, model_ds in forecast_dsets.items():
            if 'step' in model_ds.coords and selected_forecast_step in pd.to_timedelta(model_ds.step.data):
                try:
                    print(f"Processing model: {model_name}")
                    model_at_step = model_ds[variable_name].sel(step=selected_forecast_step)
                    model_interp = model_at_step.interp(latitude=common_lats, longitude=common_lons, method='linear')
                    
                    print(f"  - Interpolating ground truth for {model_name}'s timeline...")
                    model_valid_times = model_interp.base_time.data + selected_forecast_step
                    gt_for_this_model = processed_gt_da.interp(
                        time=model_valid_times,
                        method='linear'
                    ).compute(scheduler='threads')

                    gt_ready = xr.DataArray(
                        data=gt_for_this_model.data,
                        coords=model_interp.coords,
                        dims=model_interp.dims
                    )
                    
                    print(f"  - Calculating RMSE for {model_name}...")
                    error = model_interp - gt_ready
                    rmse = np.sqrt((error**2).mean(dim='base_time', skipna=True)).compute()
                    
                    rmse_data[model_name] = rmse
                    all_rmse_values.extend(rmse.values.flatten())

                    del model_at_step, model_interp, gt_for_this_model, gt_ready, error, rmse
                    gc.collect()

                except Exception as e:
                    print(f"  - An error occurred for {model_name} at step {hr_lead}h: {e}")
            else:
                print(f"Warning: Step {selected_forecast_step} not in {model_name}. Skipping model.")

        if not rmse_data:
            print(f"No RMSE data could be processed for {hr_lead} hr lead time. Skipping plot.")
            continue
        
        print("Plotting results...")
        valid_rmse_values = np.array(all_rmse_values)[~np.isnan(np.array(all_rmse_values))]
        if valid_rmse_values.size == 0:
            print(f"No valid RMSE values for {hr_lead} hr lead time. Skipping plot.")
            continue
        vmin, vmax = np.nanmin(valid_rmse_values), np.nanmax(valid_rmse_values)
        
        # Sort so that 'persistence' appears first in the plot
        sorted_model_names = sorted(rmse_data.keys(), key=lambda x: (x != 'persistence', x))
        num_models = len(rmse_data)
        if num_models == 0: continue

        fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 7), squeeze=False, sharey=True)
        axes = axes.flatten()

        for i, model_name in enumerate(sorted_model_names):
            rmse_da = rmse_data[model_name]
            ax = axes[i]
            im = rmse_da.plot.imshow(ax=ax, cmap='viridis', vmin=vmin, vmax=vmax, add_colorbar=False)
            model_overall_rmse = float(np.nanmean(rmse_da.values))
            formatted_model_rmse = f"{model_overall_rmse:.4f}"
            ax.set_title(f'{model_name.replace("_", " ").title()} RMSE\n(Overall: {formatted_model_rmse})')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude' if i == 0 else '')

        fig.subplots_adjust(right=0.85, top=0.88)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(f'RMSE of {variable_name.replace("_", " ").title()}')
        fig.suptitle(f'Spatial RMSE at {hr_lead} hr Lead Time', fontsize=16)
        
        filename = os.path.join(output_dir, f"rmse_heatmap_{hr_lead}hr_lead{suffix}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close(fig)


if __name__ == "__main__":
    
    # Define paths in a dictionary for cleaner passing
    forecast_paths = {
        "gc-finetuned": '/Datastorage/divij.khaitan_asp25/model_package_India-Finetune-100epoch/predicted.zarr',
        "gc-base": '/Datastorage/divij.khaitan_asp25/model_package_India-Finetune-100epoch/predicted_base.zarr',
        "ncep": '/Datastorage/divij.khaitan_asp25/forecasts_2022/ncep_forecasts.nc',
        "imd": '/Datastorage/divij.khaitan_asp25/forecasts_2022/imd_forecasts.nc',
        "hres": '/Datastorage/divij.khaitan_asp25/forecasts_2022/hres_forecasts.nc' # Add hres if needed
    }

    groundtruth_path = '/Datastorage/divij.khaitan_asp25/model_package_India-Finetune-100epoch/groundtruth.zarr'

    # Define the date filter parameters
    date_filter = {
        "start_date": "2022-01-01",
        "end_date": "2023-01-01",
        "midnight": False # For 00:00 and 12:00
    }

    plot_and_save_spatial_rmse_heatmaps(
        ground_truth_path=groundtruth_path,
        forecast_paths=forecast_paths,
        variable_name='2m_temperature',
        suffix='_12_hourly',
        date_filter_kwargs=date_filter
    )