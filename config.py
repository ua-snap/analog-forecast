"""Config file for analog forecasts"""

import os
from pathlib import Path

try:
    data_dir = Path(os.getenv("DATA_DIR"))
except TypeError:
    print("DATA_DIR env variable not set correctly. Must be set to path of directory containing ERA5 data.")

try:
    project_dir = Path(os.getenv("PROJECT_DIR"))
except TypeError:
    print("PROJECT_DIR env variable not set correctly. Must be set to path of the analog-forecast repository.")

# these are the params for downloading the historical data
#  (not the "recent" data for actual forecasts)
clim_cretrieve_kwargs = {   
    "product_type": "reanalysis",
    "format": "netcdf",
    # only going through 2022 right now to avoid getting "expver=5" (initial release data)
    "year": [str(year) for year in range(1959, 2022)],
    "month": [str(month).zfill(2) for month in range(1, 13)],
    "day": [str(day).zfill(2) for day in range(1, 32)],
    "time": "12:00",
    "area": [90, -180, 0, 180],
}
