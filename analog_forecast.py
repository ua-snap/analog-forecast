"""Script to look for analogs in ERA5 datasets and return RMSE and dates. 
"""

import argparse
import xarray as xr
from dask.distributed import Client
import pandas as pd
import numpy as np
#local
import luts
from config import data_dir


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
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
    
    return args.varname, args.ref_date, args.spatial_domain, args.workers


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


def find_analogs(varname, ref_date, spatial_domain, data_dir, workers):
    """Find the analogs. Starts and stops a dask cluster for the processing. 
    
    Args:
        varname (str): name of variable to search analogs based on
        ref_date (str): reference date in formate YYYY-mm-dd
        spatial_domain (str): name of the spatial domain to use
        data_dir (pathlib.PosixPath): path to the directory containing the ERA5 data files
        workers (int): number of dask workers to use for localcluster
        
    Returns:
        analogs (xarray.DataArray): data array of RMSE values and dates for 5 best analogs
    """
    # start dask cluster
    client = Client(n_workers=workers)
    
    # open connection to file
    fp = data_dir.joinpath(luts.varnames_lu[varname]["anom_filename"])
    ds = xr.open_dataset(fp)
    
    # get the bbox for search
    bbox = luts.spatial_domains[spatial_domain]["bbox"]
    
    # compute RMSE between ref_date and all preceding dates 
    #  for the specified variable and spatial domain
    rmse_da = get_rmse(ds[varname], ref_date, bbox)
    
    # sort before dropping duplicated years
    rmse_da = rmse_da.sortby(rmse_da)
    # drop duplicated years.
    # This is being done because analogs might occur in the same year as the
    #  reference date and notes from meetings with collaborators indicate that
    #  there should only be one analog per year, as was the case for the
    #  previous iteration of the algorithm.
    keep_indices = ~pd.Series(rmse_da.time.dt.year).duplicated()
    analogs = rmse_da.isel(time=keep_indices)
    # subset to first 5 analogs for now
    analogs = analogs.isel(time=slice(5))

    print("   Top 5 Analogs: ")
    for rank, date, rmse in zip(
        [1, 2, 3, 4, 5], pd.to_datetime(analogs.time.values), analogs.values
    ):
        print(f"Rank {rank}:   Date: {date:%Y-%m-%d};  RMSE: {round(rmse, 3):.3f}")
    
    ds.close()
    client.close()
    
    return analogs


def make_forecast(analogs, varname, ref_date, spatial_domain, data_dir):
    """Use a dataarray of analogs containing dates to create a composite forecast for 14 days following reference date
    
    Args:
        analogs (xarray.DataArray): data array of RMSE values and dates for 5 best analogs
        ref_date (str): reference date in formate YYYY-mm-dd
        spatial_domain (str): name of the spatial domain to use
        data_dir (pathlib.PosixPath): path to the directory containing the ERA5 data files
    
    Returns:
        forecast (xarray.DataArray): forecast values computed for the 14 days following ref_date. Computed as the mean of corresponding days following each analog. 
    """
    # get the bbox for composite mean/"forecast"
    bbox = luts.spatial_domains[spatial_domain]["bbox"]
    
    # get filepath to data file used
    fp = data_dir.joinpath(luts.varnames_lu[varname]["filename"])
    
    # susbet the data spatially as was done for analogs, then extract
    #  the next 14 days following the analog date
    with xr.open_dataset(fp) as ds:
        ds = ds.sel(
            latitude=slice(bbox[3], bbox[1]),
            longitude=slice(bbox[0], bbox[2])
        )
        arr = []
        for t in analogs.time:
            time_sl = slice(
                t + pd.to_timedelta(1, unit="d"),
                t + pd.to_timedelta(14, unit="d")
            )
            arr.append(ds[varname].sel(
                time=time_sl,

            ).values)
    # take mean over axis 0 (year/analog axis) to get the forecast
    #  will have shape of (14, n_lat, n_lon)
    composite = np.array(arr).mean(axis=0)
    
    # construct a new data array
    time = pd.date_range(
        pd.to_datetime(ref_date + " 12:00:00") + pd.to_timedelta(1, unit="d"),
        periods=14
    )
    forecast = xr.DataArray(
        data=composite,
        dims=ds[varname].dims,
        coords=dict(
            longitude=(["longitude"], ds.longitude.data),
            latitude=(["latitude"], ds.latitude.data),
            time=time,
        ),
    )
    
    return forecast

if __name__ == "__main__":
    # parse some args
    varname, ref_date, spatial_domain, workers = parse_args()
    analogs = find_analogs(varname, ref_date, spatial_domain, data_dir, workers)
    