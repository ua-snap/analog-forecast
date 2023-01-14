"""This is the script that was used to download the hourly ERA5 geopotential height at 500 hPa data for the analog forecast tool.
"""

import argparse
import logging
from pathlib import Path
import cdsapi
# local
from download_era5 import download
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", dest="download_dir", type=str, help="Output directory, where files will be downloaded to")
    args = parser.parse_args()
    download_dir = Path(args.download_dir)
    
    logging.basicConfig(level=logging.INFO)
    download(
        download_dir,
        "reanalysis-era5-pressure-levels",
        "geopotential",
        "500"
    )
