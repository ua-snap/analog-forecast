"""Worker script to profile the analog method for a certain variable and data type. Will execute the analog forecast algorithm and compute the forecast error for multiple reference dates and write results to disk.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
# local imports available with modified PYTHONPATH
import luts
from config import data_dir
from analog_forecast import make_forecast, find_analogs, spatial_subset


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--varname",
        type=str,
        help="Variable name.",
        required=True
    )
    parser.add_argument(
        "--results_file",
        type=str,
        help="path to write results to",
        required=True
    )
    parser.add_argument(
        "--use_anom",
        action=argparse.BooleanOptionalAction,
        help=f"Set this switch to use anomalies for searching for algorithms",
        default=False,
    )
    parser.add_argument(
        "-w", dest="workers", type=int, help="Number of workers to use for dask", default=8
    )
    args = parser.parse_args()
    
    return args.varname, args.use_anom, Path(args.results_file), args.workers


def load_module(fp, name):
    """Load a module by absolute path.
    
    Args:
        fp (path-like): path to module to load
        name (str): name of the module
        
    Returns:
        module object
    """
    # import modules from project
    spec = importlib.util.spec_from_file_location(name, fp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    
    return mod


def forecast_and_error(da, times, ref_date):
    """Construct a forecast using times and a reference date, and return the error values
    
    Args:
        da (xarray.DataArray): ref_date
        times (list/array-like): flat array or list of datetimes (which must be present in da) to use for constructing the forecast
        ref_date (str): date 
        
    Returns:
        the forecast error, only RMSE supported for now
    """
    forecast = make_forecast(da, times, ref_date)
    err = da.sel(time=forecast.time.values) - forecast
    
    return (err ** 2).mean(axis=(1, 2)).drop("time")


def run_analog_forecast(da, ref_date, raw_da=None):
    analogs = find_analogs(da, ref_date)
    if raw_da is not None:
        # raw_da will be supplied if da is anomaly data
        #  it will be needed for generating forecasts
        analog_error = forecast_and_error(raw_da, analogs.time.values, ref_date)
    else:
        analog_error = forecast_and_error(da, analogs.time.values, ref_date)
    
    # start a table entry
    err_df = pd.DataFrame({
        "reference_date": ref_date,
        "forecast_day_number": np.arange(14) + 1,
        "forecast_error": analog_error.values,
    })
    
    return err_df


def profile_analog_forecast(da, ref_dates, raw_da=None):
    """Profile the analog forecast for a given data array at the different ref_dates.
    Return a dataframe of results.
    """
    if raw_da is not None:
        # raw_da will be supplied if da is anomaly data
        #  it will be needed for generating forecasts
        results = [run_analog_forecast(da, date, raw_da) for date in ref_dates]
    else:
        results = [run_analog_forecast(da, date) for date in ref_dates]
    err_df = pd.concat(results)
    
    # these attributes are constant for all reference dates
    err_df["variable"] = varname
    err_df["spatial_domain"] = spatial_domain
    err_df["anomaly_search"] = use_anom
    # reorder columns
    err_df = err_df[[
        "variable", 
        "spatial_domain", 
        "anomaly_search", 
        "reference_date", 
        "forecast_day_number", 
        "forecast_error"
    ]]
    
    return err_df


def get_naive_sample_dates(all_times, naive_ref_date):
    """Constructs list of all dates to be queried in some fashion for an instance of the naive forecast"""
    analog_times = list(np.random.choice(all_times, 5, replace=False))
    naive_ref_date = pd.to_datetime(naive_ref_date + " 12:00:00").to_numpy()
    # if any sample times are closer than 15 days, re-sample
    while np.any((np.diff(sorted(analog_times + [naive_ref_date])) / (10**9 * 60 * 60 * 24)).astype(int) <= 14):
        analog_times = list(np.random.choice(all_times, 5, replace=False))
    
    all_dates = []
    for t in analog_times + [naive_ref_date]:
        all_dates.extend(pd.date_range(t, t + pd.to_timedelta(14, unit="d")))
    
    return all_dates, analog_times


def profile_naive_forecast(da, n=1000, ncpus=16):
    """Profiles the naive forecast method using a single data array with time, latitude, and longitude dimensions.
    Return a dataframe of results.
    """
    results = []
    for i in range(n):
        # significant speed-up in pooling achieved by first sub-selecting the times of interest from in-memory datarray
        #  times of interest will be the naive analog dates, the reference date, and the 14 days after all of them.
        # (not sure if the above really applies with non-Pool-based method now, but it shouldn't hurt)
        all_naive_dates, naive_analog_dates = get_naive_sample_dates(sub_da.time.values[:-15], naive_ref_date)
        results.append(forecast_and_error(sub_da.sel(time=all_naive_dates), naive_analog_dates, naive_ref_date))
    
    sim_rmse = xr.concat(results, pd.Index(range(n), name="sim"))

    err_df = pd.DataFrame({
        "variable": da.name,
        "spatial_domain": spatial_domain,
        "anomaly_search": use_anom,
        "forecast_day_number": np.arange(14) + 1,
        "naive_2.5": sim_rmse.reduce(np.percentile, dim="sim", q=2.5),
        "naive_50": sim_rmse.reduce(np.percentile, dim="sim", q=50),
        "naive_97.5": sim_rmse.reduce(np.percentile, dim="sim", q=97.5),
    })
    
    return err_df


if __name__ == "__main__":
    varname, use_anom, results_fp, workers = parse_args()
    
    # set up the reference dates we will be using
    ref_dates = ["2004-10-11", "2004-10-18", "2005-09-22", "2013-11-06", "2004-05-09", "2015-11-09", "2015-11-23"]
    # ok and the reference dates we actually want are the dates which precede these dates by 3 and 5 days,
    #  so that the forecasts start 3 and 5 days ahead of these reference dates
    ref_dates = [
        (pd.to_datetime(date) - pd.to_timedelta(3, unit="d")).strftime("%Y-%m-%d")
        for date in ref_dates
    ] + [
        (pd.to_datetime(date) - pd.to_timedelta(5, unit="d")).strftime("%Y-%m-%d")
        for date in ref_dates
    ]
    # arbitrary reference date for naive forecasts
    naive_ref_date = ref_dates[0]
    
    # load the data - strategy is to just load all the data, then iterate over domains
    fp_lu_key = {True: "anom_filename", False: "filename"}[use_anom]
    fp = data_dir.joinpath(luts.varnames_lu[varname][fp_lu_key])
    print("Reading in search data")
    ds = xr.load_dataset(fp)
    if use_anom:
        print("Reading in raw data for forecasting with anomaly analogs")
        # also will load raw data if anomaly search is used
        raw_ds = xr.load_dataset(data_dir.joinpath(luts.varnames_lu[varname]["filename"]))
    else:
        raw_ds = None
     
    analog_results = []
    naive_results = []
    
    for spatial_domain in luts.spatial_domains:
        print(f"Working on {spatial_domain}")
        bbox = luts.spatial_domains[spatial_domain]["bbox"]
        sub_da = spatial_subset(ds[varname], bbox)
        if raw_ds is not None:
            raw_da = spatial_subset(raw_ds[varname], bbox)
            tmp_result = profile_analog_forecast(sub_da, ref_dates, raw_da)
        else:
            raw_da = None
            tmp_result = profile_analog_forecast(sub_da, ref_dates)
        # profile the analog forecast by computing for all dates
        analog_results.append(tmp_result)
        # profile the naive forecast
        naive_results.append(profile_naive_forecast(sub_da))
        
    naive_df = pd.concat(naive_results)
    analog_df = pd.concat(analog_results)
    
    analog_df.round(3).to_csv(results_fp, index=False)
    naive_df.round(3).to_csv(results_fp.name.replace(".csv", "_naive.csv"), index=False)
