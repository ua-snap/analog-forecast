"""Script to look for analogs in ERA5 datasets and return RMSE and dates. 

TO-DO: compute forecasts
"""


import argparse
import xarray as xr
from dask.distributed import Client
import pandas as pd
import numpy as np
#local
import luts
from config import data_dir


# TO-DO: fix for single year! particularly SST, all analogs are most recent days
def get_rmse(da, ref_date, bbox):
    """Get the RMSE between da at ref_time and every other preceding time slice. Calls compute() to process with dask.
    
    Args:
        da (xarray.DataArray): dataarray of ERA5 data, with time, latitude, longitude axes
        ref_date (str): reference date to compare all other preceding time slices against
        bbox (tuple): bounding box for subdomain selection
        
    Returns:
        rmse_da (xarra.DataArray): data array with an RMSE computed for each date preceding ref_date
    """
    sub_da = da.sel(latitude=slice(bbox[3], bbox[1]), longitude=slice(bbox[0], bbox[2]))
    ref_da = sub_da.sel(time=ref_date).squeeze()
    end_search_date = pd.to_datetime(ref_date) - pd.to_timedelta(1, unit="d")
    search_da = sub_da.sel(time=slice(da.time.values[0], end_search_date))
    rmse_da = np.sqrt(((ref_da - search_da) ** 2).mean(axis=(0, 1)))
    
    # calls compute to run the computation with dask
    return rmse_da.compute()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("-v", dest="varname", type=str, help="Variable name, either 'tp' or 'pev'")
    parser.add_argument(
        "-v",
        dest="varname",
        type=str,
        help=(
            "Variable name to look for analogs. "
            "Options are t2m (temperature), sst (sea surface temperature), msl (mean sea level pressure)"
        ),
        required=True
    )
    parser.add_argument(
        "-d", dest="ref_date", type=str, help="Date to use for analog search", required=True
    )
    parser.add_argument(
        "-s",
        dest="spatial_domain",
        type=str,
        help=f"Spatial domain to use for analog search and forecast. Options are {', '.join(list(luts.spatial_domains.keys()))}",
        default="alaska",
    )
    parser.add_argument(
        "-w", dest="workers", type=int, help="Number of workers to use for dask", default=8
    )
    args = parser.parse_args()
    varname = args.varname
    ref_date = args.ref_date

    fp = data_dir.joinpath(luts.varnames_lu[varname]["filename"])

    # start dask cluster
    client = Client(n_workers=args.workers)
    # open connection to file
    ds = xr.open_dataset(fp, chunks="auto")

    bbox = luts.spatial_domains[args.spatial_domain]["bbox"]

    rmse_da = get_rmse(ds[varname], ref_date, bbox)

    # get the 5 best analogs
    analogs = rmse_da.sortby(rmse_da).isel(time=slice(5))

    print("   Top 5 Analogs: ")
    for rank, date, rmse in zip(
        [1, 2, 3, 4, 5], pd.to_datetime(analogs.time.values), analogs.values
    ):
        print(f"Rank {rank}:   Date: {date:%Y-%m-%d};  RMSE: {round(rmse, 3):.3f}")

    client.close()
