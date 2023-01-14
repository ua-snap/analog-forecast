"""Module for assisting with ERA5 download scripts"""

import logging
import cdsapi


def download(download_dir, dataset, varnames, pressure_level=None):
    """Download ERA5 data at 12:00 for each day from 1959 to 2021"""
    # trying to see if this can be done all in one request (might be too many "elements")
    logging.info(f"Downloading hourly ERA5 data to {download_dir}")
    c = cdsapi.Client()
    
    download_dict = {   
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": varnames,
        # only going through 2022 right now to avoid getting "expver=5" (initial release data)
        "year": [str(year) for year in range(1959, 2022)],
        "month": [str(month).zfill(2) for month in range(1, 13)],
        "day": [str(day).zfill(2) for day in range(1, 32)],
        "time": "12:00",
        "area": [90, -180, 0, 180],
    }
    if pressure_level:
        download_dict["pressure_level"] = pressure_level

    for varname in varnames:
        c.retrieve(
            dataset,
            download_dict,
            download_dir.joinpath(f"era5_{varname}_hour12_1959_2021.nc"),
        )

    return
