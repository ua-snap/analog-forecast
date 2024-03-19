"""Main worker script for running the analog forecast. This script can be run from the command line like so:

python analog_forecast.py -v <variable name -d <reference date> 

The following other command line options are available:
  -sd <search domain>: search domain, default is alaska (see README for more options)
  -fd <forecast domain>: forecast domain, default is alaska (see README for more options)
  --use-anom: boolean flag to use anomalies for identifying analogs (default is False)
  -n <number of analogs>: number of analogs to pull for generating forecast (default is 5)
  -w <number of dask workers>: number of workers to use in dask (default is 8)

Example usage with all args:
  python analog_forecast.py -v sst -d 2021-12-01 -sd alaska -fd northern_hs --use-anom -n 4 -w 12

If run from the command line like so, forecasts will be written to a forecasts/ folder created from the execution location. 
"""

import argparse
from datetime import datetime
from pathlib import Path
import xarray as xr
from dask.distributed import Client
import pandas as pd
import numpy as np

# local
import luts
from config import data_dir
from scripts.download_era5 import run_download


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
        required=True,
    )
    parser.add_argument(
        "-d",
        dest="ref_date",
        type=str,
        help="Date to use for analog search (YYYY-MM-DD format)",
        required=True,
    )
    parser.add_argument(
        "-sd",
        dest="search_domain",
        type=str,
        help=f"Spatial domain to use for analog search. Options are {', '.join(list(luts.spatial_domains.keys()))}",
        default="alaska",
    )
    parser.add_argument(
        "-fd",
        dest="forecast_domain",
        type=str,
        help=f"Spatial domain to use for analog forecast. Options are {', '.join(list(luts.spatial_domains.keys()))}",
        default="alaska",
    )
    parser.add_argument(
        "--use-anom",
        dest="use_anom",
        action="store_true",
        help=f"Use anomalies for search.",
    )
    parser.add_argument(
        "-n", dest="n_analogs", type=int, help="Number of analogs to use", default=5
    )
    parser.add_argument(
        "-w",
        dest="workers",
        type=int,
        help="Number of workers to use for dask",
        default=8,
    )
    args = parser.parse_args()

    return (
        args.varname,
        args.ref_date,
        args.search_domain,
        args.forecast_domain,
        args.use_anom,
        args.n_analogs,
        args.workers,
    )


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
        da_w = da.sel(
            latitude=slice(bbox[3], bbox[1]), longitude=slice(-180, (bbox[2] - 360))
        )
        da_e = da.sel(latitude=slice(bbox[3], bbox[1]), longitude=slice(bbox[0], 180))
        da_w = da_w.assign_coords(longitude=da_w.longitude.values + 360)
        sub_da = xr.combine_by_coords([da_w, da_e])[varname]
    else:
        sub_da = da.sel(
            latitude=slice(bbox[3], bbox[1]), longitude=slice(bbox[0], bbox[2])
        )

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

    # susbet the data spatially as was done for analogs
    with xr.open_dataset(fp) as ds:
        sub_da = spatial_subset(ds[varname], bbox)

    return sub_da


def prune_search_da(search_da, ref_date, window_size):
    """Prune the search DataArray to consist only of dates that are a day-of-year within a n-day window centered on the day-of-year of the reference date

    Args:
        search_da (xarray.DataArray): ERA5 dataArray, with time, latitude, longitude axes
        ref_date (str): reference date to compare all other preceding time slices against
        window_size (int): size of inclusion window.

    Returns:
        pruned_da (xarray.DataArray): ERA5 DataArray consisting only of dates that are a day-of-year within a n-day window centered on the day-of-year of the reference date
    """
    half_window = round(window_size / 2)
    ref_dtime = pd.to_datetime(ref_date + " 12:00:00")
    ref_window = pd.date_range(
        ref_dtime - pd.to_timedelta(half_window, "d"),
        ref_dtime + pd.to_timedelta(half_window - 1, "d"),
    )
    ref_doys = xr.DataArray(ref_window).dt.dayofyear.values
    pruned_da = search_da.where(search_da.time.dt.dayofyear.isin(ref_doys), drop=True)

    return pruned_da


def get_search_da(da, ref_date, window, window_size=90):
    """Temporally subset a DataArray to only the desired timestamps for searching for analogs

    Args:
        da (xarray.DataArray): ERA5 dataArray, with time, latitude, longitude axes
        ref_date (str): reference date to compare all other preceding time slices against
        window (str): option for omitting dates from the search window.
          'precede' for the search window to stop 14 days short of the reference date;
            (Note, this potentially severely limits forcasts from earlier in the historical period!)
          'any' for only omitting the 14 days before/after the reference date, and the last 14 days of available data.
        window_size (int): size of inclusion window.

    Returns:
        search_da ((xarray.DataArray): data array to be searched for analogs
    """
    if window == "precede":
        # don't want to accept analogs that are within 15 days of the reference date
        end_search_date = pd.to_datetime(ref_date) - pd.to_timedelta(15, unit="d")
        search_da = da.sel(time=slice(da.time.values[0], end_search_date))
    elif window == "any":
        # all timestamps considered except those which cannot allow 14 day forecasts
        #  (i.e. the last 14 timestamps and the 14 dates before and after the reference date)
        end_search_date = da.time.values[-1] - pd.to_timedelta(15, unit="d")
        search_da = da.sel(time=slice(da.time.values[0], end_search_date))

        ref_dtime = pd.to_datetime(ref_date + " 12:00:00")
        ref_window = pd.date_range(
            ref_dtime - pd.to_timedelta(15, "d"), ref_dtime + pd.to_timedelta(14, "d")
        )
        search_da = search_da.where(~search_da.time.isin(ref_window), drop=True)

    # prune search_da to only include days of year within seasonal (90 day) window
    search_da = prune_search_da(search_da, ref_date, window_size=window_size)

    return search_da


def rmse(da, ref_da):
    """Compute the RMSE between each time slice in da and ref_date

    Args:
        da (xarray.DataArray): data array with time, latitude, and longitude dims for computing RMSE over the spatial dimensions
        ref_da (xarray.DataArray): reference data array with only spatial axes (2d grid) to compute RMSE against

    Returns:
        rmse_da (xarrya.DataArray): data array where each time step is the RMSE between da at that same time step and ref_da
    """
    try:
        rmse_arr = np.sqrt(
            np.apply_over_axes(np.nanmean, (da - ref_da) ** 2, axes=[1, 2]).squeeze()
        )
    except MemoryError:
        print("Encountered MemoryError. Retrying with dask")
        da = da.chunk(time=1)
        sq_err = (da - ref_da) ** 2
        rmse_arr = np.sqrt(np.nanmean(sq_err, axis=(1, 2)))

    rmse_da = xr.DataArray(
        data=rmse_arr,
        dims=["time"],
        coords=dict(
            time=da.time.values,
        ),
    )

    return rmse_da


def run_rmse_over_time(da, window, ref_da=None, ref_date=None):
    """Get the RMSE between da at ref_time and every other preceding time slice. Calls compute() to process with dask.

    Args:
        da (xarray.DataArray): ERA5 dataArray, with time, latitude, longitude axes
        window (str): method for temporal search, passed to get_search_da
        ref_da (xarray.DataArray): DataArray of ERA5 data at reference date. Provide if this is already available
        ref_date (str): reference date in format YYYY-mm-dd, if ref_da not provided

    Returns:
        rmse_da (xarra.DataArray): data array with an RMSE computed for each date preceding ref_date
    """
    if ref_da is None:
        if ref_date is None:
            exit("Must provide either reference date if available in da, or ref_da")
        else:
            ref_da = da.sel(time=ref_date).squeeze()
    else:
        if ref_date is not None:
            exit(
                "Both ref_da and ref_date were provided. Ignoring ref_date in favor of ref_da"
            )
        ref_date = pd.to_datetime(ref_da.time.values[0]).strftime("%Y-%m-%d")
        # need to make sure ref_da has no time dimension for rmse()
        ref_da = ref_da.squeeze()

    search_da = get_search_da(da, ref_date, window)
    rmse_da = rmse(search_da, ref_da)

    return rmse_da


def take_analogs(error_da, buffer, n_analogs=5):
    """Take the top n times from error_da ('top' == smallest error) with constraints on how close in time these top times may be.

    Args:
        error_da (xarray.DataArray): error values indexed by time
        buffer (int): number of days to buffer each top time for excluding subsequent times
        n_analogs (int): number of analogs to keep

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
            # check to see if analog is within any windows in analog_buffer_ranges.
            # If not, "keep" it, then add a new buffer window to avoid for subsequent analogs.
            analog_buffer_ranges.append(pd.date_range(t - td_buffer, t + td_buffer))
            analog_times.append(t)
        if len(analog_buffer_ranges) == n_analogs:
            break

    analogs = error_da.sel(time=analog_times)

    return analogs


def find_analogs(
    search_da, print_analogs=False, ref_da=None, ref_date=None, n_analogs=5
):
    """Find the analogs.

    Args:
        search_da (xarray.DataArray): data array of ERA5 data (likely already subset to area of interest)
        print_analogs (bool): print the top analogs and scores
        ref_da (xarray.DataArray): DataArray of ERA5 data at reference date. Provide if this is already available
        ref_date (str): reference date in format YYYY-mm-dd, if ref_da not provided
        n_analogs (int): number of analogs to search for.

    Returns:
        analogs (xarray.DataArray): data array of RMSE values and dates for 5 best analogs
    """
    # compute RMSE between ref_date and all dates in search_da
    # currently the window option of "any" is hard-coded, which just means it will
    #  look for analogs from all possible data besides what can't be used to construct analogs
    #  (i.e. dates that would result in forecast inputs being pulled
    #  from those we are actually trying to forecast)
    rmse_da = run_rmse_over_time(
        search_da, window="any", ref_da=ref_da, ref_date=ref_date
    )
    varname = search_da.name

    # impose restrictions on temporal proximity for analog based on variable.
    #  if SST, then we use an exclusion buffer of 6 months, atmospheric vars use 30 days
    if varname in ["t2m", "msl", "z"]:
        buffer = 30
    elif varname in ["sst"]:
        buffer = 180
    analogs = take_analogs(rmse_da, buffer, n_analogs=n_analogs)

    if print_analogs:
        print(f"   Top {n_analogs} Analogs: ")
        for rank, date, rmse in zip(
            np.arange(n_analogs) + 1,
            pd.to_datetime(analogs.time.values),
            analogs.values,
        ):
            print(f"Rank {rank}:   Date: {date:%Y-%m-%d};  RMSE: {round(rmse, 3):.3f}")

    assert ~any(np.isnan(analogs))

    return analogs


def make_forecast(sub_da, times, ref_date):
    """Use a dataarray of analogs with dates to create a composite forecast for 14 days following reference date

    Args:
        sub_da (xarray.DataArray): data array of ERA5 data for variable and region of interest
        times (list): list of times to generate forecast from
        ref_date (str): reference date in formate YYYY-mm-dd

    Returns:
        forecast (xarray.DataArray): forecast values computed for the 14 days following ref_date. Computed as the mean of corresponding days following each analog.
    """
    # iterate over the analog time values (n of them, where n is the number of analogs)
    #  and extract the 14 following days for each, creating an array with shape (n, 14, n_lat, n_lon)
    arr = []
    for t in times:
        time_sl = slice(
            t + pd.to_timedelta(1, unit="d"), t + pd.to_timedelta(14, unit="d")
        )
        arr.append(sub_da.sel(time=time_sl).values)

    # take mean over axis 0 (year/analog axis) to get the forecast
    #  will have shape of (14, n_lat, n_lon)
    composite = np.array(arr).mean(axis=0)

    # construct a new data array
    time = pd.date_range(
        pd.to_datetime(ref_date + " 12:00:00") + pd.to_timedelta(1, unit="d"),
        periods=14,
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


def run_forecast(
    varname,
    search_domain,
    forecast_domain,
    ref_date,
    n_analogs,
    use_anom,
    print_analogs=True,
    return_error=False,
):
    """Highest level function to run the analog forecast for a supplied reference date. Will download ERA5 data if reference date is not found in historical archive.

    Args:
        varname (str): name of variable for forecast (and search)
        search_domain (str): name of spatial domain to use for search
        forecast_domain (str): name of spatial domain to use for forecast
        ref_date (str): reference date for which to construct forecast of following 14 days from, in format YYYY-mm-dd
        n_analogs (int): number of analogs to use for constructing forecast
        use_anom (bool): use anomalies for search
        print_analogs (bool): print out the analogs
        return_error (bool): include the forecast error array (raw array minus forecast) in tuple with forecast array

    Returns:
        forecast (xarray.DataArray): forecast data array consisting of forecasts for the 14 dates following the reference date and with spatial extent matching the forecast domain
    """
    ref_date = pd.to_datetime(ref_date + "T12:00:00")
    # read in the historical archive of specified the variable for searching
    search_da = read_subset_era5(search_domain, data_dir, varname, use_anom=use_anom)

    if use_anom:
        # if using anomalies, will need the raw data as well for forecasting
        raw_da = read_subset_era5(search_domain, data_dir, varname, use_anom=False)
    else:
        # if not using anomalies, search data is already the raw data
        raw_da = search_da

    # download data for reference date if it is not part of the current historical archive
    if ref_date not in search_da.time[:-15]:
        print("Reference date not found in historical archive, downloading...")
        bbox = list(luts.spatial_domains[search_domain]["bbox"])
        era5_dataset_name = luts.varnames_lu[varname]["era5_dataset_name"]
        era5_long_name = luts.varnames_lu[varname]["era5_long_name"]
        out_paths = run_download(
            bbox, ref_date, data_dir, era5_dataset_name, era5_long_name
        )
        ref_da = xr.load_dataset(out_paths[0])[varname]

        if use_anom:
            # if using anomalies and have downloaded data, will need to compute anomalies for the reference date
            # compute the climatology from the raw data for the day of year of the reference date
            clim_da = raw_da.sel(
                time=raw_da.time[raw_da.dayofyear == ref_date.dayofyear]
            ).mean(dim="time")
            ref_da = ref_da - clim_da
    else:
        ref_da = search_da.sel(time=[ref_date])

    # find the analogs
    analogs = find_analogs(
        search_da, print_analogs=print_analogs, ref_da=ref_da, n_analogs=n_analogs
    )

    if forecast_domain != search_domain:
        # if forecast domain and search domain are different, read in a new subset of ERA5 for the forecast domain
        raw_da = read_subset_era5(forecast_domain, data_dir, varname, use_anom=False)
    else:
        # otherwise, use the raw data for constructing the forecast
        raw_da = raw_da

    forecast = make_forecast(raw_da, analogs.time.values, ref_date.strftime("%Y-%m-%d"))
    forecast.attrs["analogs"] = list(analogs.time.dt.strftime("%Y-%m-%d").values)

    if return_error:
        err = raw_da.sel(time=forecast.time.values) - forecast
        err.name = "error"
        return forecast, err
    else:
        return forecast


if __name__ == "__main__":
    # parse some args
    (
        varname,
        ref_date,
        search_domain,
        forecast_domain,
        use_anom,
        n_analogs,
        workers,
    ) = parse_args()

    # start dask cluster
    client = Client(n_workers=workers)
    # run analog search
    # get the ERA5 data for searching
    forecast, err = run_forecast(
        varname,
        search_domain,
        forecast_domain,
        ref_date,
        n_analogs,
        use_anom,
        print_analogs=True,
        return_error=True,
    )

    forecast.name = varname
    forecast.attrs.update(
        {
            "variable": varname,
            "search_domain": search_domain,
            "forecast_domain": forecast_domain,
            "reference_date": ref_date,
            "anomaly_based_search": "true" if use_anom else "false",
        }
    )
    forecast_ds = forecast.to_dataset()

    err.attrs = {"name": "Analog forecast error", "error_type": "rmse"}
    forecast_ds = forecast_ds.assign(error=err)

    out_anom_suffix = "_anom" if use_anom else ""
    out_fp = Path("forecasts").joinpath(
        f"{varname}_{search_domain}_{forecast_domain}_{ref_date}{out_anom_suffix}.nc"
    )
    out_fp.parent.mkdir(exist_ok=True)
    forecast_ds.to_netcdf(out_fp)

    # close cluster
    client.close()
