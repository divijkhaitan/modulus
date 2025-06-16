import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from visualise_spatial_errors import filter_dates

def plot_total_rmse_by_lead_time(
    ground_truth_ds,
    forecast_datasets,
    variable_name="2m_temperature",
    output_dir="total_rmse_plots",
    suffix=""
):
    """
    Generates and saves a line plot showing the total RMSE for each forecast
    lead time, comparing all forecast datasets and a persistence benchmark to
    the ground truth. The total RMSE for a given lead time is calculated by

    averaging the spatial RMSE over latitude and longitude.

    Args:
        ground_truth_ds (xr.Dataset): Dataset for ground truth.
        forecast_datasets (dict): Dictionary of forecast datasets.
        variable_name (str): The variable to plot (default: "2m_temperature").
        output_dir (str): Directory to save the plot (default: "total_rmse_plots").
        suffix (str): An optional suffix for the output filename.
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    common_lats = np.linspace(-90, 90, 721)
    common_lons = np.linspace(0, 359.75, 1440)

    processed_forecast_datasets = {}
    for ds_name, ds in forecast_datasets.items():
        if variable_name not in ds.data_vars:
            print(f"Warning: Variable '{variable_name}' not found in '{ds_name}'. Skipping.")
            continue
        interp_ds = ds[variable_name].interp(
            {'latitude': common_lats, 'longitude': common_lons},
            method='linear',
            kwargs={'fill_value': None}
        )
        if 'time' in interp_ds.coords and 'base_time' not in interp_ds.coords:
            interp_ds = interp_ds.rename({'time': 'base_time'})
        if 'base_time' not in interp_ds.coords:
            print(f"Warning: Dataset '{ds_name}' needs 'base_time' or 'time'. Skipping.")
            continue
        processed_forecast_datasets[ds_name] = interp_ds

    if variable_name not in ground_truth_ds.data_vars:
        raise ValueError(f"Variable '{variable_name}' not in ground truth dataset.")
    processed_gt_da = ground_truth_ds[variable_name].interp(
        {'latitude': common_lats, 'longitude': common_lons},
        method='linear',
        kwargs={'fill_value': None}
    )

    if 'base_time' in processed_gt_da.dims and 'step' in processed_gt_da.coords:
        first_step = pd.to_timedelta(processed_gt_da.step.data[0])
        print(f"Note: Ground truth has 'base_time' and 'step'. Using first available step ({first_step}) as observation time.")
        processed_gt_da = processed_gt_da.isel(step=0, drop=True)
        processed_gt_da = processed_gt_da.rename({'base_time': 'time'})
    elif 'base_time' in processed_gt_da.dims and 'time' not in processed_gt_da.dims:
        print("Note: Renaming 'base_time' to 'time' in ground truth.")
        processed_gt_da = processed_gt_da.rename({'base_time': 'time'})
    elif 'time' not in processed_gt_da.dims:
        raise ValueError("Ground truth must have a 'time' or 'base_time' dimension.")

    all_steps_values = sorted(list(set(
        pd.Timedelta(s) for ds in processed_forecast_datasets.values()
        if 'step' in ds.coords for s in ds.step.data
        if pd.Timedelta(s) != pd.Timedelta(0)
    )))

    if not all_steps_values:
        print("No valid forecast lead times found to plot. Exiting.")
        return

    model_names = list(processed_forecast_datasets.keys())
    total_rmse_results = {model_name: [] for model_name in model_names + ['persistence']}
    lead_times_hours = []

    for selected_forecast_step in all_steps_values:
        step_for_sel = pd.to_timedelta(selected_forecast_step)
        hr_lead = int(step_for_sel / np.timedelta64(1, 'h'))
        lead_times_hours.append(hr_lead)
        print(f"Calculating total RMSE for lead time: {hr_lead} hours...")

        # --- Persistence Benchmark Calculation ---
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
                
                gt_persistence_forecast['time'] = gt_at_valid_time['time']
                
                aligned_gt, aligned_persistence = xr.align(gt_at_valid_time, gt_persistence_forecast, join='inner')
                
                error_persistence = aligned_gt - aligned_persistence
                spatial_rmse = np.sqrt((error_persistence**2).mean(dim='time', skipna=True))
                total_rmse = spatial_rmse.mean(dim=['latitude', 'longitude'], skipna=True).compute().item()
                total_rmse_results['persistence'].append(total_rmse)
            except Exception as e:
                print(f"Error in persistence calculation for step {hr_lead}h: {e}")
                total_rmse_results['persistence'].append(np.nan)
        else:
            total_rmse_results['persistence'].append(np.nan)

        # --- Model Forecasts Calculation ---
        for model_name, model_ds in processed_forecast_datasets.items():
            if 'step' in model_ds.coords and step_for_sel in model_ds.step.data:
                try:
                    model_at_step = model_ds.sel(step=step_for_sel)
                    forecast_valid_times = model_at_step['base_time'].data + step_for_sel
                    gt_aligned = processed_gt_da.interp(
                        time=forecast_valid_times,
                        method='linear',
                        kwargs={'fill_value': np.nan}
                    )
                    gt_aligned_for_subtraction = gt_aligned.rename({'time': 'base_time'})
                    aligned_model, aligned_gt = xr.align(
                        model_at_step, gt_aligned_for_subtraction, join='inner'
                    )
                    error = aligned_model - aligned_gt
                    spatial_rmse = np.sqrt((error**2).mean(dim='base_time', skipna=True))
                    total_rmse = spatial_rmse.mean(dim=['latitude', 'longitude'], skipna=True).compute().item()
                    total_rmse_results[model_name].append(total_rmse)
                except Exception as e:
                    print(f"Error for {model_name} at step {hr_lead}h: {e}")
                    total_rmse_results[model_name].append(np.nan)
            else:
                total_rmse_results[model_name].append(np.nan)

    plt.figure(figsize=(12, 7))
    for model_name, rmse_values in total_rmse_results.items():
        valid_indices = [i for i, val in enumerate(rmse_values) if pd.notna(val)]
        if valid_indices:
            plt.plot(
                [lead_times_hours[i] for i in valid_indices],
                [rmse_values[i] for i in valid_indices],
                marker='o', linestyle='-', label=model_name
            )

    plt.xlabel('Forecast Lead Time (hours)', fontsize=12)
    plt.ylabel(f'Total RMSE of {variable_name.replace("_", " ").title()}', fontsize=12)
    plt.title(f'Total RMSE vs. Lead Time for {variable_name.replace("_", " ").title()}', fontsize=14)
    plt.xticks(sorted(list(set(lead_times_hours))))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    filename = os.path.join(output_dir, f"total_rmse_line_plot_{variable_name}{suffix}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


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

    plot_total_rmse_by_lead_time(groundtruth_filtered, {"gc-finetuned": predictions_filtered, "gc-base": base_filtered, "ncep":ncep_filtered, "imd":imd_filtered}, '2m_temperature', suffix='all_world')