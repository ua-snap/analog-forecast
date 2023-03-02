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


def spatial_subset(da, bbox):
    """Subset a DataArray spatially. Allows using a bbox that crosses the antimeridian. 
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


def get_search_da(da, ref_date, window):
    """Temporally subset a DataArray to only the desired timestamps for searching for analogs, based on method
    
    Args:
        da (xarray.DataArray): ERA5 dataArray, with time, latitude, longitude axes
        ref_date (str): reference date to compare all other preceding time slices against
        temporal_search_method (str): method for temporal search, passed to get_search_da
        
    Returns:
        search_da ((xarray.DataArray): data array to be searched for analogs
    """
    if window == "precede":
        # don't want to accept analogs that are withing 15 days of the reference date
        end_search_date = pd.to_datetime(ref_date) - pd.to_timedelta(15, unit="d")
        search_da = da.sel(time=slice(da.time.values[0], end_search_date))
    elif window == "any": 
        # all timestamps considered except those which cannot allow 14 day forecasts
        #  (i.e. the last 14 timestamps and the 14 dates before and after the reference date)
        end_search_date = da.time.values[-1] - pd.to_timedelta(15, unit="d")
        search_da = da.sel(time=slice(da.time.values[0], end_search_date))
        
        ref_dtime = pd.to_datetime(ref_date + " 12:00:00")
        ref_window = pd.date_range(
            ref_dtime - pd.to_timedelta(15, "d"),
            ref_dtime + pd.to_timedelta(14, "d")
        )
        search_da = search_da.where(~search_da.time.isin(ref_window), drop=True)
        
    return search_da


def rmse(da, ref_da):
    """Compute the RMSE between each time slice in da and ref_date
    
    Args:
        da (xarray.DataArray): data array with time, latitude, and longitude dims for computing RMSE over the spatial dimensions
        ref_da (xarray.DataArray): reference data array with only spatial axes (2d grid) to compute RMSE against
        
    Returns:
        rmse_da (xarrya.DataArray): data array where each time step is the RMSE between da at that same time step and ref_da
    """
    # rmse_da = np.sqrt(((da - ref_da) ** 2).mean(axis=(1, 2)))
    
    rmse_arr = np.sqrt(np.apply_over_axes(np.nanmean, (da - ref_da) ** 2, axes=[1,2]).squeeze())
    rmse_da = xr.DataArray(
        data=rmse_arr,
        dims=["time"],
        coords=dict(
            time=da.time.values,
        ),
    )
    
    return rmse_da


def run_rmse_over_time(da, ref_date, window):
    """Get the RMSE between da at ref_time and every other preceding time slice. Calls compute() to process with dask.
    
    Args:
        da (xarray.DataArray): ERA5 dataArray, with time, latitude, longitude axes
        ref_date (str): reference date to compare all other preceding time slices against
        window (str): method for temporal search, passed to get_search_da
        
    Returns:
        rmse_da (xarra.DataArray): data array with an RMSE computed for each date preceding ref_date
    """
    ref_da = da.sel(time=ref_date).squeeze()
    search_da = get_search_da(da, ref_date, window)
    rmse_da = rmse(search_da, ref_da)
    
    return rmse_da


def take_analogs(error_da, buffer, ref_date, n=5):
    """Take the top n times from error_da ('top' == smallest error) with constraints on how close in time these top times may be.
    
    Args:
        error_da (xarray.DataArray): error values indexed by time
        buffer (int): number of days to buffer each top time for excluding subsequent times
        ref_date (str): 
        n (int): number of analogs to keep
        
    Returns:
        analogs (xarray.DataArray): top n analogs taken from error_da
    """
    # sort ascending
    error_da = error_da.sortby(error_da)
    # iterate over list of top analogs and create a window that others must not overlap, lest they be discarded
    analog_buffer_ranges = []
    analog_times = []
    td_buffer = pd.to_timedelta(buffer, "d")
    for t in error_da.time.values:
        if not any([t in dr for dr in analog_buffer_ranges]):
            analog_buffer_ranges.append(pd.date_range(t - td_buffer, t + td_buffer))
            analog_times.append(t)
        if len(analog_buffer_ranges) == 5:
            break

    analogs = error_da.sel(time=analog_times)

    return analogs


def find_analogs(da, ref_date, print_analogs=False):
    """Find the analogs.
    
    Args:
        da (xarray.DataArray): data array of ERA5 data (likely already subset to area of interest)
        ref_date (str): reference date in formate YYYY-mm-dd
        print_analogs (bool): print the top 5 analogs and scores
        
    Returns:
        analogs (xarray.DataArray): data array of RMSE values and dates for 5 best analogs
    """
    # compute RMSE between ref_date and all preceding dates 
    #  for the specified variable and spatial domain
    rmse_da = run_rmse_over_time(da, ref_date, "any")
    varname = da.name
    
    # subset to first 5 analogs for now
    # impose restrictions on temporal proximity for analog based on variable.
    #  if SST, then we use an exclusion buffer of 6 months, atmospheric vars use 30 days
    if varname in ["t2m", "msl", "z"]:
        buffer = 30
    elif varname in ["sst"]:
        buffer = 180
    analogs = take_analogs(rmse_da, buffer, ref_date, 5)
    
    if print_analogs:
        print("   Top 5 Analogs: ")
        for rank, date, rmse in zip(
            [1, 2, 3, 4, 5], pd.to_datetime(analogs.time.values), analogs.values
        ):
            print(f"Rank {rank}:   Date: {date:%Y-%m-%d};  RMSE: {round(rmse, 3):.3f}")
    
    assert ~any(np.isnan(analogs))
    
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
    # get the ERA5 data for searching
    sub_da = read_subset_era5(spatial_domain, data_dir, varname, use_anom)
    analogs = find_analogs(da, ref_date, print_analogs=True)
    # close cluster
    client.close()
