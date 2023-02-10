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


def run_rmse_over_time(da, ref_date):
    """Get the RMSE between da at ref_time and every other preceding time slice. Calls compute() to process with dask.
    
    Args:
        da (xarray.DataArray): dataarray of ERA5 data, with time, latitude, longitude axes
        ref_date (str): reference date to compare all other preceding time slices against
        
    Returns:
        rmse_da (xarra.DataArray): data array with an RMSE computed for each date preceding ref_date
    """
    ref_da = da.sel(time=ref_date).squeeze()
    end_search_date = pd.to_datetime(ref_date) - pd.to_timedelta(1, unit="d")
    search_da = da.sel(time=slice(da.time.values[0], end_search_date))
    rmse_da = np.sqrt(((ref_da - search_da) ** 2).mean(axis=(0, 1)))
    
    # calls compute to run the computation with dask
    return rmse_da.compute()


def spatial_subset(da, bbox):
    """Subset the dataset. Allows using a bbox that crosses the antimeridian. 
    Trying to reassing the longitude coordinate before subsetting was causing intense slowdowns with xarray for some reason. So this function actually divides a bbox into two chunks, loads the data for each, and then combines back into a single dataset on the [0, 360) scale. 
    
    Args:
        da (xarray.DataArray): dataset with longitude coordinate variable on [-180, 180) scale
        bbox (tuple): bounding box for subdomain selection where 0 <= bbox[0] < bbox[2] < 360
    
    Returns:
        sub_da (xarray.DataArray): dataset with longitude coordinate variable on [0, 360) scale
    """
    varname = da.name
    if 0 <= bbox[0] < bbox[2] < 360:
        da_w = da.sel(latitude=slice(bbox[3], bbox[1]), longitude=slice(-180, (bbox[2] - 360))).load()
        da_e = da.sel(latitude=slice(bbox[3], bbox[1]), longitude=slice(bbox[0], 180)).load()
        da_w = da_w.assign_coords(longitude=da_w.longitude.values + 360)
        sub_da = xr.combine_by_coords([da_w, da_e])[varname]
    else:  
        sub_da = da.sel(latitude=slice(bbox[3], bbox[1]), longitude=slice(bbox[0], bbox[2]))
    
    return sub_da


def read_subset_era5(spatial_domain, data_dir, varname, use_anom):
    """Opens a connection to the ERA5 dataset for the variable of interest and subsets to the specified spatial region.
    
    Args:
        spatial_domain (str): name of the spatial domain to subset to
        data_dir (pathlib.PosixPath): path to directory containing ERA5 data
        varname (str): name of the variable being used
        use_anom (bool): whether or not to use anomalies
        
    Returns:
        sub_da (xarray.DataArray): data array of ERA5 data subset to the spatial domain of interest
    """
    # get the bbox
    bbox = luts.spatial_domains[spatial_domain]["bbox"]

    # get filepath to data file used
    # These will be for forecasts only so we want the real data, not anomaly
    # open connection to file
    if use_anom:
        fp_lu_key = "anom_filename"
    else:
        fp_lu_key = "filename"
    fp = data_dir.joinpath(luts.varnames_lu[varname][fp_lu_key])

    # susbet the data spatially as was done for analogs, then extract
    #  the next 14 days following the analog date
    with xr.open_dataset(fp) as ds:        
        sub_da = spatial_subset(ds[varname], bbox)
        
    return sub_da


def find_analogs(varname, ref_date, spatial_domain, data_dir, workers, use_anom):
    """Find the analogs.
    
    Args:
        varname (str): name of variable to search analogs based on
        ref_date (str): reference date in formate YYYY-mm-dd
        spatial_domain (str): name of the spatial domain to use
        data_dir (pathlib.PosixPath): path to the directory containing the ERA5 data files
        use_anom (bool): whether or not to use anomalies for analog search
        
    Returns:
        analogs (xarray.DataArray): data array of RMSE values and dates for 5 best analogs
    """
    # get the ERA5 data for searching
    sub_da = read_subset_era5(spatial_domain, data_dir, varname, use_anom)
    
    # compute RMSE between ref_date and all preceding dates 
    #  for the specified variable and spatial domain
    rmse_da = run_rmse_over_time(sub_da, ref_date)
    
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
    
    return analogs


def make_forecast(sub_da, times, ref_date):
    """Use a dataarray of analogs containing dates to create a composite forecast for 14 days following reference date
    
    Args:
        sub_da (xarray.DataArray): data array of ERA5 data for variable and region of interest
        times (list): list of times to generate forecast from
        ref_date (str): reference date in formate YYYY-mm-dd
    
    Returns:
        forecast (xarray.DataArray): forecast values computed for the 14 days following ref_date. Computed as the mean of corresponding days following each analog. 
    """
    arr = []
    for t in times:
        time_sl = slice(
            t + pd.to_timedelta(1, unit="d"),
            t + pd.to_timedelta(14, unit="d")
        )
        arr.append(sub_da.sel(time=time_sl).values)

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
        dims=sub_da.dims,
        coords=dict(
            longitude=(["longitude"], sub_da.longitude.data),
            latitude=(["latitude"], sub_da.latitude.data),
            time=time,
        ),
    )
    
    return forecast

if __name__ == "__main__":
    # parse some args
    varname, ref_date, spatial_domain, workers = parse_args()
    
    # start dask cluster
    client = Client(n_workers=workers, dashboard_address="localhost:33338")
    # run analog search
    analogs = find_analogs(varname, ref_date, spatial_domain, data_dir)
    # close cluster
    client.close()