import xarray as xr
import glob
import os
import numpy as np

GRIB_FOLDER = '/Datastorage/divij.khaitan_asp25/hres_forecasts_2022'
OUTPUT_NC_FILE_IMD = '/Datastorage/divij.khaitan_asp25/forecasts_2022/imd_forecasts.nc'
OUTPUT_NC_FILE_NCEP = '/Datastorage/divij.khaitan_asp25/forecasts_2022/ncep_forecasts.nc'
OUTPUT_NC_FILE_HRES = '/Datastorage/divij.khaitan_asp25/forecasts_2022/hres_forecasts.nc'

def main():
    """
    Main function to find GRIB files and process them efficiently.
    """
    grib_files = sorted(glob.glob(os.path.join(GRIB_FOLDER, '*.grib')))

    if not grib_files:
        print(f"No GRIB files found in {GRIB_FOLDER}. Please check the path and file extension.")
        return
    
    print(f"Found {len(grib_files)} GRIB files to process.")

    # print("\n--- Starting IMD data processing (Memory-Efficient Method) ---")
    # try:
    #     print("Creating virtual dataset for IMD...")
    #     combined_imd_ds = xr.open_mfdataset(
    #         grib_files,
    #         engine='cfgrib',
    #         combine='by_coords',
    #         parallel=True, 
    #         backend_kwargs={'filter_by_keys': {'numberOfPoints': 4503000}, 'indexpath': ''}
    #     )
        
    #     print(f"Virtual IMD dataset created. Saving to: {OUTPUT_NC_FILE_IMD}")
    #     new_lats = np.linspace(-90, 90, 721)
    #     new_lons = np.linspace(0, 359.75, 1440)
    #     combined_imd_ds = combined_imd_ds.interp({'latitude': new_lats, 'longitude': new_lons}, 
    #                                 method='linear',
    #                                 kwargs={'fill_value': None})
    #     combined_imd_ds.to_netcdf(OUTPUT_NC_FILE_IMD)
    #     print("--- IMD data processing complete. ---\n")

    # except Exception as e:
    #     print(f"An error occurred during IMD processing: {e}")
    #     print("This could be due to a corrupted file or inconsistent data structures among files.")


    # print("\n--- Starting NCEP data processing (Memory-Efficient Method) ---")
    # try:
    #     print("Creating virtual dataset for NCEP...")
    #     combined_ncep_ds = xr.open_mfdataset(
    #         grib_files,
    #         engine='cfgrib',
    #         combine='by_coords',
    #         parallel=True,
    #         backend_kwargs={'filter_by_keys': {'numberOfPoints': 259920}, 'indexpath': ''}
    #     )

    #     print(f"Virtual NCEP dataset created. Saving to: {OUTPUT_NC_FILE_NCEP}")
    #     combined_ncep_ds.to_netcdf(OUTPUT_NC_FILE_NCEP)
    #     print("--- NCEP data processing complete. ---")

    # except Exception as e:
    #     print(f"An error occurred during NCEP processing: {e}")
    #     print("This could be due to a corrupted file or inconsistent data structures among files.")

    print("\n--- Starting HRES data processing (Memory-Efficient Method) ---")
    # try:
    print("Creating virtual dataset for HRES...")
    combined_hres_ds = xr.open_mfdataset(
        grib_files,
        engine='cfgrib',
        combine='by_coords',
        parallel=True,
    )
    combined_hres_ds = combined_hres_ds.set_index(values=('latitude', 'longitude'))
    combined_hres_ds = combined_hres_ds.unstack('values')
    new_lats = np.linspace(-90, 90, 721)
    new_lons = np.linspace(0, 359.75, 1440)
    print(combined_hres_ds)
    combined_hres_ds = combined_hres_ds.interp({'latitude': new_lats, 'longitude': new_lons}, 
                                method='linear',
                                kwargs={'fill_value': None})
    
    print(f"Virtual HRES dataset created. Saving to: {OUTPUT_NC_FILE_HRES}")
    combined_hres_ds.to_netcdf(OUTPUT_NC_FILE_HRES)
    print("--- HRES data processing complete. ---")

    # except Exception as e:
    #     print(f"An error occurred during HRES processing: {e}")
    #     print("This could be due to a corrupted file or inconsistent data structures among files.")

if __name__ == "__main__":
    main()
