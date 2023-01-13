"""This is the script that was used to download the hourly ERA5 temperature, sea level pressure, and sea surface temperature data for the analog forecast tool.
"""

import argparse
import logging
from pathlib import Path
import cdsapi


def download(download_dir):
    """Download potential evaporation for the climatology period of 1981-2020"""
    # trying to see if this can be done all in one request (might be too many "elements")
    logging.info(f"Downloading hourly ERA5 data to {download_dir}")
    c = cdsapi.Client()
    for varname in ["2m_temperature", "mean_sea_level_pressure", "sea_surface_temperature"]:
        c.retrieve(
            "reanalysis-era5-single-levels",
            {   
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": varname,
                # only going through 2022 right now to avoid getting "expver=5" (initial release data)
                "year": [str(year) for year in range(1959, 2022)],
                "month": [str(month).zfill(2) for month in range(1, 13)],
                "day": [str(day).zfill(2) for day in range(1, 32)],
                "time": "12:00",
                "area": [90, -180, 0, 180],
            },
            download_dir.joinpath(f"era5_{varname}_hour12_1959_2021.nc"),
        )

    return
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("-v", dest="varname", type=str, help="Variable name, either 'tp' or 'pev'")
    parser.add_argument("-d", dest="download_dir", type=str, help="Output directory, where files will be downloaded to")
    args = parser.parse_args()
    # varname = args.varname
    # long_name = long_name_lu[varname]
    download_dir = Path(args.download_dir)
    
    logging.basicConfig(level=logging.INFO)
    download(download_dir)
