import xarray as xr
import glob
import os
import numpy as np

GRIB_FOLDER = '/Datastorage/divij.khaitan_asp25/hres_forecasts_2022'
TEMP_NC_FOLDER = '/Datastorage/divij.khaitan_asp25/forecasts_2022/hres_temp_processed' # Directory for intermediate files
OUTPUT_NC_FILE_HRES = '/Datastorage/divij.khaitan_asp25/forecasts_2022/hres_forecasts.nc'

def main():
    """
    Processes HRES GRIB files, extracts a specific variable, regrids it,
    saves intermediate NetCDF files, and combines them into a final file.
    """
    grib_files = sorted(glob.glob(os.path.join(GRIB_FOLDER, '*.grib')))

    if not grib_files:
        print(f"No GRIB files found in {GRIB_FOLDER}. Please check the path.")
        return

    print(f"Found {len(grib_files)} GRIB files to process.")

    print(f"\n--- Starting HRES individual file processing ---")
    os.makedirs(TEMP_NC_FOLDER, exist_ok=True)
    
    new_lats = np.linspace(90, -90, 721)
    new_lons = np.linspace(0, 359.75, 1440)

    # for i, grib_file in enumerate(grib_files):
    #     print(f"Processing file {i+1}/{len(grib_files)}: {os.path.basename(grib_file)}")
    #     try:
    #         temp_output_path = os.path.join(
    #             TEMP_NC_FOLDER,
    #             os.path.basename(grib_file).replace('.grib', '.nc')
    #         )

    #         if os.path.exists(temp_output_path):
    #             print("  -> Already processed. Skipping.")
    #             continue

    #         # --- START OF CORRECTION ---
    #         # Define backend_kwargs to filter for a specific variable.
    #         # This is crucial for files with multiple conflicting variables.
    #         cfgrib_kwargs = {'filter_by_keys': {'shortName': '2t'}}
    #         ds = xr.open_dataset(grib_file, engine='cfgrib', backend_kwargs=cfgrib_kwargs)
            
    #         # Check if dataset is empty after filtering
    #         if not ds.data_vars:
    #             print(f"  -> WARNING: No variable with shortName '2t' found in {os.path.basename(grib_file)}. Skipping.")
    #             continue

    #         # **CORRECTED LOGIC**: Only unstack if the data is in the 1D 'values' format.
    #         # After filtering, cfgrib can often parse the 2D grid directly.
    #         if 'values' in ds.dims:
    #             print("  -> Data is in 1D 'values' format. Reshaping to 2D grid.")
    #             ds = ds.set_index(values=('latitude', 'longitude'))
    #             ds = ds.unstack('values')
    #         else:
    #             print("  -> Data is already in a 2D grid. No reshaping needed.")
    #         # --- END OF CORRECTION ---
            
    #         # The regridding logic remains the same
    #         print("  -> Regridding to target grid.")
    #         ds_regridded = ds.interp(
    #             latitude=new_lats, 
    #             longitude=new_lons,
    #             method='linear'
    #         )
            
    #         print(f"  -> Saving temporary file to: {temp_output_path}")
    #         ds_regridded.to_netcdf(temp_output_path)

    #     except Exception as e:
    #         print(f"  -> ERROR processing {os.path.basename(grib_file)}: {e}")
    #         print("  -> Skipping this file.")
    #         continue
    
    print("\n--- Individual file processing complete. ---")

    print("\n--- Starting final combination step ---")
    try:
        processed_files = sorted(glob.glob(os.path.join(TEMP_NC_FOLDER, '*.nc')))

        if not processed_files:
            print("No processed files found to combine.")
            return

        print(f"Combining {len(processed_files)} processed NetCDF files sequentially...")

        # --- REPLACEMENT FOR open_mfdataset ---
        # 1. Create a list to hold each individual dataset.
        list_of_datasets = []

        # 2. Loop through each file path, open it as a dataset, and append to the list.
        for nc_file in processed_files:
            try:
                ds = xr.open_dataset(nc_file)
                list_of_datasets.append(ds)
            except Exception as e:
                print(f"  -> WARNING: Could not open {os.path.basename(nc_file)}. Error: {e}. Skipping.")
                continue
        
        if not list_of_datasets:
            print("No valid datasets could be opened for combination. Aborting.")
            return

        # 3. Concatenate the list of datasets into a single dataset along the 'time' dimension.
        #    xarray automatically aligns the other coordinates (latitude, longitude).
        print("All files opened. Concatenating into a single dataset...")
        final_ds = xr.concat(list_of_datasets, dim='time')

        # 4. (Optional but recommended) Sort by the time coordinate to ensure chronological order.
        final_ds = final_ds.sortby('time')
        # --- END OF REPLACEMENT ---

        print(f"Dataset created. Saving final file to: {OUTPUT_NC_FILE_HRES}")
        # Note: The final dataset is now fully in memory. Saving it writes the complete data.
        final_ds.to_netcdf(OUTPUT_NC_FILE_HRES)
        
        print("--- HRES data processing complete. ---")

    except Exception as e:
        print(f"An error occurred during the final combination step: {e}")

if __name__ == "__main__":
    main()