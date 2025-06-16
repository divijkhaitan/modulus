import xarray as xr
import glob
import os
import numpy as np

GRIB_FOLDER = '/Datastorage/divij.khaitan_asp25/hres_forecasts_2022'
TEMP_NC_FOLDER = '/Datastorage/divij.khaitan_asp25/forecasts_2022/hres_temp_processed' # Directory for intermediate files
OUTPUT_NC_FILE_HRES = '/Datastorage/divij.khaitan_asp25/forecasts_2022/hres_forecasts.nc'

def main():
    grib_files = sorted(glob.glob(os.path.join(GRIB_FOLDER, '*.grib')))

    if not grib_files:
        print(f"No GRIB files found in {GRIB_FOLDER}. Please check the path.")
        return

    print(f"Found {len(grib_files)} GRIB files to process.")

    print(f"\n--- Starting HRES individual file processing ---")
    os.makedirs(TEMP_NC_FOLDER, exist_ok=True)
    
    # Define the target grid once
    new_lats = np.linspace(-90, 90, 721)
    new_lons = np.linspace(0, 359.75, 1440)

    for i, grib_file in enumerate(grib_files):
        print(f"Processing file {i+1}/{len(grib_files)}: {os.path.basename(grib_file)}")
        try:
            temp_output_path = os.path.join(
                TEMP_NC_FOLDER,
                os.path.basename(grib_file).replace('.grib', '.nc')
            )

            if os.path.exists(temp_output_path):
                print("  -> Already processed. Skipping.")
                continue

            ds = xr.open_dataset(grib_file, engine='cfgrib')

            ds = ds.set_index(values=('latitude', 'longitude'))
            ds = ds.unstack('values')
            ds = ds.interp(
                {'latitude': new_lats, 'longitude': new_lons},
                method='linear',
                kwargs={'fill_value': None} 
            )
            
            ds.to_netcdf(temp_output_path)
            print(f"  -> Saved temporary file to: {temp_output_path}")

        except Exception as e:
            print(f"  -> ERROR processing {os.path.basename(grib_file)}: {e}")
            print("  -> Skipping this file.")
            continue
    
    print("\n--- Individual file processing complete. ---")

    print("\n--- Starting final combination step ---")
    try:
        processed_files = sorted(glob.glob(os.path.join(TEMP_NC_FOLDER, '*.nc')))

        if not processed_files:
            print("No processed files found to combine.")
            return

        print(f"Combining {len(processed_files)} processed NetCDF files...")
        
        final_ds = xr.open_mfdataset(
            processed_files,
            combine='by_coords',
            parallel=True
        )

        print(f"Virtual dataset created. Saving final file to: {OUTPUT_NC_FILE_HRES}")
        final_ds.to_netcdf(OUTPUT_NC_FILE_HRES)
        print("--- HRES data processing complete. ---")

    except Exception as e:
        print(f"An error occurred during the final combination step: {e}")

if __name__ == "__main__":
    main()