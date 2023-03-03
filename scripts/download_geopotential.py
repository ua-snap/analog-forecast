"""This is the script that was used to download the hourly ERA5 geopotential height at 500 hPadown data for the analog forecast tool.
"""

import argparse
import logging
from pathlib import Path
import cdsapi
# local
from download_era5 import download
from config import clim_cretrieve_kwargs
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", dest="download_dir", type=str, help="Output directory, where files will be downloaded to")
    args = parser.parse_args()
    download_dir = Path(args.download_dir)
    
    logging.basicConfig(level=logging.INFO)
    clim_cretrieve_kwargs.update({"pressure_level": "500"})
    
    download(
        download_dir=download_dir,
        dataset="reanalysis-era5-pressure-levels",
        varnames=["geopotential"],
        cretrieve_kwargs=clim_cretrieve_kwargs,
        fn_suffix="hour12_1959_2021"
    )
