import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import os

def filter_dates(ds, start_date="2022-01-01", end_date="2023-01-01", midnight=False):
    base_times = pd.to_datetime(ds['base_time'].values)
    if midnight:
        mask = ((base_times.time == pd.to_datetime("00:00").time()) | (base_times.time == pd.to_datetime("12:00").time())) & \
                        (base_times >= pd.to_datetime(start_date)) & \
                        (base_times <= pd.to_datetime(end_date))
        return ds.sel(base_time=base_times[mask])
    else:
        mask = (base_times >= pd.to_datetime(start_date)) & \
                (base_times <= pd.to_datetime(end_date))
        return ds.sel(base_time=base_times[mask])
    
def plot_and_save_spatial_rmse_heatmaps(
    ground_truth_ds,
    forecast_datasets,
    variable_name="2m_temperature",
    output_dir="rmse_heatmaps",
    suffix=''
):
    """
    Generates and saves spatial heatmaps of RMSE for each forecast lead time
    (excluding 0h lead time), comparing all forecast datasets and a persistence
    benchmark to the ground truth. This version includes a critical fix for
    correctly aligning ground truth time.

    Args:
        ground_truth_ds (xr.Dataset): Dataset for ground truth.
        forecast_datasets (dict): Dictionary of forecast datasets.
        variable_name (str): The variable to plot (default: "2m_temperature").
        output_dir (str): Directory to save heatmaps (default: "rmse_heatmaps").
        suffix (str): An optional suffix to add to the output filename.
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    common_lats = np.linspace(7.5, 37.5, 61)
    common_lons = np.linspace(67.5, 97.5, 61)

    # Process forecast datasets
    processed_forecast_datasets = {}
    for ds_name, ds in forecast_datasets.items():
        if variable_name not in ds.data_vars:
            print(f"Warning: Var '{variable_name}' not in '{ds_name}'. Skipping.")
            continue
        interp_ds = ds[variable_name].interp(
            {'latitude': common_lats, 'longitude': common_lons},
            method='linear',
            kwargs={'fill_value': None}
        )
        if 'time' in interp_ds.coords and 'base_time' not in interp_ds.coords:
            interp_ds = interp_ds.rename({'time': 'base_time'})
        if 'base_time' not in interp_ds.coords:
            raise ValueError(f"Dataset '{ds_name}' must have 'base_time' or 'time' coordinate.")
        processed_forecast_datasets[ds_name] = interp_ds

    # Process ground truth dataset
    if variable_name not in ground_truth_ds.data_vars:
        raise ValueError(f"Variable '{variable_name}' not found in ground truth dataset.")
    processed_gt_da = ground_truth_ds[variable_name].interp(
        {'latitude': common_lats, 'longitude': common_lons},
        method='linear',
        kwargs={'fill_value': None}
    )

    # --- START: CORRECTED GROUND TRUTH TIME HANDLING ---
    if 'base_time' in processed_gt_da.dims and 'step' in processed_gt_da.coords:
        first_step_value = pd.to_timedelta(processed_gt_da.step.data[0])
        print(f"Note: GT has 'base_time' and 'step'. Using first step ({first_step_value}) as the observation.")
        
        processed_gt_da_at_step = processed_gt_da.sel(step=first_step_value)
        
        # CRITICAL FIX: Calculate the actual valid time for the ground truth observation.
        valid_time_coords = processed_gt_da_at_step.base_time + first_step_value
        
        # Assign correct valid times to a new 'time' coordinate and make it the dimension.
        processed_gt_da = processed_gt_da_at_step.assign_coords(time=valid_time_coords).swap_dims({'base_time': 'time'}).drop_vars(['step', 'base_time'], errors='ignore')

    elif 'base_time' in processed_gt_da.dims and 'time' not in processed_gt_da.dims:
        print("Note: Renaming 'base_time' to 'time' in ground truth dataset.")
        processed_gt_da = processed_gt_da.rename({'base_time': 'time'})
    elif 'time' not in processed_gt_da.dims:
        raise ValueError("Ground truth dataset must have a 'time' or ('base_time', 'step') dimension.")
    # --- END: CORRECTED GROUND TRUTH TIME HANDLING ---

    all_steps_values = sorted(list(set(
        pd.Timedelta(s) for ds in processed_forecast_datasets.values()
        if 'step' in ds.coords for s in ds.step.data
        if pd.Timedelta(s) != pd.Timedelta(0)
    )))

    if not all_steps_values:
        print("No forecast lead times found to plot after filtering 0h. Exiting.")
        return

    for selected_forecast_step in all_steps_values:
        step_for_sel = pd.to_timedelta(selected_forecast_step)
        hr_lead = int(step_for_sel / np.timedelta64(1, 'h'))
        print(f"Processing lead time: {hr_lead} hours...")

        rmse_data = {}
        all_rmse_values = []

        # --- START: PERSISTENCE BENCHMARK CALCULATION ---
        # Find a reference set of base_times from any model available at the current step
        ref_base_times = None
        for model_ds in processed_forecast_datasets.values():
            if 'step' in model_ds.coords and step_for_sel in model_ds.step.data:
                ref_base_times = model_ds.sel(step=step_for_sel)['base_time'].data
                break
        
        if ref_base_times is not None:
            try:
                valid_times = ref_base_times + step_for_sel
                persistence_times = ref_base_times

                gt_at_valid_time = processed_gt_da.interp(time=valid_times, method='linear')
                gt_persistence_forecast = processed_gt_da.interp(time=persistence_times, method='linear')
                
                # Align persistence forecast time coord with the valid time coord for subtraction
                gt_persistence_forecast['time'] = gt_at_valid_time['time']
                
                aligned_gt, aligned_persistence = xr.align(gt_at_valid_time, gt_persistence_forecast, join='inner')
                
                error_persistence = aligned_gt - aligned_persistence
                rmse_persistence = np.sqrt((error_persistence**2).mean(dim='time', skipna=True))
                
                rmse_data['persistence'] = rmse_persistence
                all_rmse_values.extend(rmse_persistence.values.flatten())
                print("Persistence RMSE calculated.")
            except Exception as e:
                print(f"Could not calculate persistence benchmark for step {hr_lead}h: {e}")
        
        for model_name, model_ds in processed_forecast_datasets.items():
            if 'step' in model_ds.coords and step_for_sel in model_ds.step.data:
                try:
                    model_at_step = model_ds.sel(step=step_for_sel)
                    forecast_valid_times = model_at_step['base_time'].data + step_for_sel

                    gt_aligned_to_forecast_time = processed_gt_da.interp(
                        time=forecast_valid_times, method='linear', kwargs={'fill_value': np.nan}
                    )
                    gt_ready_for_subtraction = gt_aligned_to_forecast_time.rename({'time': 'base_time'})
                    aligned_model, aligned_gt = xr.align(model_at_step, gt_ready_for_subtraction, join='inner')
                    
                    if aligned_model['base_time'].size == 0:
                         print(f"Warning: No overlapping time points for {model_name} at step {hr_lead}h. Skipping.")
                         continue

                    error = aligned_model - aligned_gt
                    rmse = np.sqrt((error**2).mean(dim='base_time', skipna=True))
                    
                    rmse_data[model_name] = rmse
                    all_rmse_values.extend(rmse.values.flatten())
                except Exception as e:
                    print(f"An error occurred for {model_name} at step {selected_forecast_step}: {e}")
            else:
                print(f"Warning: Step {selected_forecast_step} not in {model_name}. Skipping model for this lead time.")

        if not rmse_data:
            print(f"No RMSE data could be processed for {hr_lead} hr lead time. Skipping plot.")
            continue

        valid_rmse_values = np.array(all_rmse_values)[~np.isnan(np.array(all_rmse_values))]
        if valid_rmse_values.size == 0:
            print(f"No valid RMSE values for {hr_lead} hr lead time. Skipping plot.")
            continue
        vmin, vmax = np.nanmin(valid_rmse_values), np.nanmax(valid_rmse_values)
        
        # Sort model names to have a consistent order, e.g., persistence first
        sorted_model_names = sorted(rmse_data.keys(), key=lambda x: (x != 'persistence', x))

        num_models = len(rmse_data)
        fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 7), squeeze=False, sharey=True)
        axes = axes.flatten()

        for i, model_name in enumerate(sorted_model_names):
            rmse_da = rmse_data[model_name]
            ax = axes[i]
            im = rmse_da.plot.imshow(
                ax=ax, cmap='viridis', vmin=vmin, vmax=vmax, add_colorbar=False
            )
            model_overall_rmse = float(np.nanmean(rmse_da.values))
            formatted_model_rmse = f"{model_overall_rmse:.4f}"

            ax.set_title(f'{model_name} RMSE\n(Overall: {formatted_model_rmse})')
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

    groundtruth = xr.open_zarr('/Datastorage/divij.khaitan_asp25/model_package_India-Finetune-100epoch/groundtruth.zarr')
    groundtruth.coords['step'] = groundtruth.coords['step'] + pd.to_timedelta(6, unit='h')
    groundtruth_filtered = filter_dates(groundtruth)

    predictions = xr.open_zarr('/Datastorage/divij.khaitan_asp25/model_package_India-Finetune-100epoch/predicted.zarr')
    predictions.coords['step'] = predictions.coords['step'] + pd.to_timedelta(6, unit='h')
    predictions_filtered = filter_dates(predictions)

    base = xr.open_zarr('/Datastorage/divij.khaitan_asp25/model_package_India-Finetune-100epoch/predicted_base.zarr')
    base.coords['step'] = base.coords['step'] + pd.to_timedelta(6, unit='h')
    base_filtered = filter_dates(base)

    imd = xr.load_dataset('/Datastorage/divij.khaitan_asp25/forecasts_2022/imd_forecasts.nc')
    rename_dict = {'time': 'base_time', 't2m': '2m_temperature', 'tp': 'total_precipitation'}

    imd = imd.rename(rename_dict)

    ncep = xr.load_dataset('/Datastorage/divij.khaitan_asp25/forecasts_2022/ncep_forecasts.nc')
    rename_dict = {'time': 'base_time', 't2m': '2m_temperature', 'tp': 'total_precipitation'}

    ncep = ncep.rename(rename_dict)

    coords_to_drop = ['number', 'heightAboveGround', 'surface', 'valid_time']
    coords_to_drop = [coord for coord in coords_to_drop if coord in ncep.coords]
    if coords_to_drop:
        ncep = ncep.drop_vars(coords_to_drop)
        imd = imd.drop_vars(coords_to_drop)
    ncep_filtered = filter_dates(ncep)
    imd_filtered = filter_dates(imd)

    # plot_and_save_spatial_rmse_heatmaps(groundtruth, {"gc-finetuned": predictions, "gc-base": base, "ncep":ncep}, '2m_temperature', suffix='')

    plot_and_save_spatial_rmse_heatmaps(groundtruth_filtered, {"gc-finetuned": predictions_filtered, "gc-base": base_filtered, "ncep":ncep_filtered, "imd":imd_filtered}, '2m_temperature', suffix='_12_hourly')