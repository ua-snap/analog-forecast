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


def forecast_and_error(da, times, ref_date):
    """Construct a forecast using timestamps and a reference date, and return the error values
    
    Args:
        da (xarray.DataArray): ref_date
        times (list/array-like): flat array or list of datetimes (which must be present in da) to use for constructing the forecast
        ref_date (str): date in format YYYY-mm-dd
        
    Returns:
        the forecast error, only RMSE supported for now
    """
    forecast = make_forecast(da, times, ref_date)
    err = da.sel(time=forecast.time.values) - forecast
    
    return np.sqrt((err ** 2).mean(axis=(1, 2)).drop("time"))


def run_analog_forecast(search_da, ref_date, raw_da):
    """Run the analog forecast for a given reference date. Differs from the "run_forecast" function in the top-level analog_forecast.py module in that ref date must be within the provided data array, and the output is structured for results tables
    
    Args:
        search_da (xarray.DataArray): data array of ERA5 data covering the spatial domain used for searching for analogs
        ref_date (str): reference date in format YYYY-mm-dd
        raw_da (xarray.DataArray): data array of ERA5 data for generating the forecast from. This should be subset to the forecast domain of interest.
        
    Returns:
        tuple of (err_df, analogs_df): err_df is pandas.DataFrame of error results, analogs_df contains analog scores for top analogs
    """
    analogs = find_analogs(search_da, ref_date=ref_date)
    analog_error = forecast_and_error(raw_da, analogs.time.values, ref_date)
    
    # start a table entry
    err_df = pd.DataFrame({
        "reference_date": ref_date,
        "forecast_day_number": np.arange(14) + 1,
        "forecast_error": analog_error.values,
    })
    
    # table entry for analog dates identified
    analogs_df = pd.DataFrame({
        "reference_date": ref_date,
        "analog_date": pd.to_datetime(analogs.time.values).strftime("%Y-%m-%d"),
        "analog_score": analogs.values
    })
    
    return err_df, analogs_df


def profile_analog_forecast(search_da, ref_dates, raw_da):
    """Profile the analog forecast for a given set of reference dates using the supplied data array, which is already subsetted to a particular spatial domain.
    
    Args:
        search_da (xarray.DataArray): data array of ERA5 data covering the spatial domain used for searching for analogs
        ref_date (str): reference date in format YYYY-mm-dd
        raw_da (xarray.DataArray): data array of ERA5 data for generating the forecast. This should be subset to the forecast domain of interest.
        
    Returns:
        two dataframes of results
    """
    results = [run_analog_forecast(search_da, date, raw_da) for date in ref_dates]

    # forecast error dataframes are first in tuple
    err_df = pd.concat([r[0] for r in results])
    # dataframe of analog dates is second part
    analogs_df = pd.concat([r[1] for r in results])
    
    # these attributes are constant for all reference dates
    err_df["variable"] = varname
    err_df["search_domain"] = search_da.attrs["spatial_domain"]
    err_df["forecast_domain"] = raw_da.attrs["spatial_domain"]
    err_df["anomaly_search"] = use_anom
    analogs_df["variable"] = varname
    analogs_df["search_domain"] = search_da.attrs["spatial_domain"]
    analogs_df["forecast_domain"] = raw_da.attrs["spatial_domain"]
    analogs_df["anomaly_search"] = use_anom
    
    # reorder columns
    err_df = err_df[[
        "variable", 
        "search_domain", 
        "forecast_domain",
        "anomaly_search", 
        "reference_date", 
        "forecast_day_number", 
        "forecast_error"
    ]]
    analogs_df = analogs_df[[
        "variable", 
        "search_domain",
        "forecast_domain",
        "anomaly_search", 
        "reference_date", 
        "analog_date",
        "analog_score",
    ]]
    
    return err_df, analogs_df


def get_naive_sample_dates(all_times, ref_date):
    """Constructs list of all dates to be used for a single naive forecast and error. 
    
    Args:
        all_times (list): list of all times available in the hsitorical archive
        ref_date (str): reference date in format YYYY-mm-dd
        
    Returns:
        tuple of (all_dates, analog_times): all_dates is a list of the analog dates and the subsequent dates of each to be used in the forecast, analog_times is a list of only the analog dates
    """
    # first, limit all_times to not include any of the last n values within the forecast
    #  length (in days) as those would have forecast times that extend beyond what is in all_times
    forecast_length = 14
    all_times = all_times[:-(forecast_length + 1)]
    
    # limit pool of all potential times for analogs to be within 3-month window centered on
    #  day-of-year of reference date
    # iterate over all available years and use the same month-day as ref_dt to 
    #  construct acceptance window of +/- 45 days, and accumulate boolean series
    #  corresponding to all_times which essentially will be reduced via "or"
    #  operation over the year axis.
    ref_dt = pd.to_datetime(ref_date + " 12:00:00")
    td_offset = pd.to_timedelta(45, "D")
    all_times = pd.Series(all_times)
    keep_bool = []
    
    for year in np.unique(pd.Series(all_times).dt.year):
        tmp_ref_dt = pd.to_datetime(f"{year}-{ref_dt.month}-{ref_dt.day}")
        keep_bool.append(
            all_times.between(
                tmp_ref_dt - (td_offset + pd.to_timedelta(1, "D")), 
                tmp_ref_dt + td_offset
            )
        )
    keep_times = all_times[np.array(keep_bool).sum(axis=0).astype(bool)]
    
    # construct an exclusion window around ref_date, based on size of forecast which is 14 days
    exclude = keep_times.between(
        ref_dt - pd.to_timedelta(forecast_length + 2, "D"),
        ref_dt + pd.to_timedelta(forecast_length + 1, "D")
    )
    keep_times[~exclude]
    
    # choose 5 times at random
    analog_times = list(np.random.choice(keep_times, 5, replace=False))
    
    all_dates = []
    for t in analog_times + [ref_dt]:
        all_dates.extend(pd.date_range(t, t + pd.to_timedelta(forecast_length, "D")))
    
    return all_dates, analog_times


def profile_naive_forecast(raw_da, ref_dates, n=1000):
    """Profiles the naive forecast method using a single data array with time, latitude, and longitude dimensions.
    Return a dataframe of results.
    
    Don't need to worry about search v forecast domains because search for analogs is randomized
    """
    err_df_rows = []
    for ref_date in ref_dates:
        results = []
        for i in range(n):
            # significant speed-up in pooling achieved by first sub-selecting the times of interest from in-memory datarray
            #  times of interest will be the naive analog dates, the reference date, and the 14 days after all of them.
            # (not sure if the above really applies with non-Pool-based method now, but it shouldn't hurt)
            all_naive_dates, naive_analog_dates = get_naive_sample_dates(raw_da.time.values, ref_date)
            # duplicated entries in da cause trouble - drop them
            all_naive_dates = list(pd.Series(sorted(all_naive_dates)).drop_duplicates())
            results.append(
                forecast_and_error(raw_da.sel(time=all_naive_dates), naive_analog_dates, ref_date)
            )
                
        sim_rmse = xr.concat(results, pd.Index(range(n), name="sim"))

        ref_err_df = pd.DataFrame({
            "variable": raw_da.name,
            "forecast_domain": raw_da.attrs["spatial_domain"],
            "reference_date": ref_date,
            "forecast_day_number": np.arange(14) + 1,
            "naive_2.5": sim_rmse.reduce(np.percentile, dim="sim", q=2.5),
            "naive_50": sim_rmse.reduce(np.percentile, dim="sim", q=50),
            "naive_97.5": sim_rmse.reduce(np.percentile, dim="sim", q=97.5),
        })
        
        err_df_rows.append(ref_err_df)
        
    err_df = pd.concat(err_df_rows)
    
    return err_df


if __name__ == "__main__":
    varname, use_anom, results_fp, workers = parse_args()
    
    # set up the reference dates we will be using
    ref_dates = ["2004-10-11", "2004-10-18", "2005-09-22", "2013-11-06", "2004-05-09", "2015-11-09", "2015-11-23", "2011-11-09"]
    # ok and the reference dates we actually want are the dates which precede these dates by 3 and 5 days,
    #  so that the forecasts start 3 and 5 days ahead of these reference dates
    ref_dates = [
        (pd.to_datetime(date) - pd.to_timedelta(3, unit="d")).strftime("%Y-%m-%d")
        for date in ref_dates
    ] + [
        (pd.to_datetime(date) - pd.to_timedelta(5, unit="d")).strftime("%Y-%m-%d")
        for date in ref_dates
    ]
    
    # load the data - strategy is to just load all the data, then iterate over domains
    fp_lu_key = {True: "anom_filename", False: "filename"}[use_anom]
    fp = data_dir.joinpath(luts.varnames_lu[varname][fp_lu_key])
    print("Reading in search data", flush=True)
    ds = xr.load_dataset(fp)
    if use_anom:
        print("Reading in raw data for forecasting with anomaly analogs", flush=True)
        # also will load raw data if anomaly search is used
        raw_ds = xr.load_dataset(data_dir.joinpath(luts.varnames_lu[varname]["filename"]))
    else:
        raw_ds = None
     
    analog_results = []
    dates_results = []
    
    #  we only need to profile the naive forecast for standard (non-anomaly-based) search
    #   because results should be identical
    if not use_anom:
        naive_results = []
    
    # okay our only forecast domain will be alaska for now
    forecast_domain = "alaska"
    for search_domain in luts.spatial_domains:
        print(f"Working on search domain of {search_domain}, forecast domain of {forecast_domain}", flush=True)
        search_bbox = luts.spatial_domains[search_domain]["bbox"]
        forecast_bbox = luts.spatial_domains[forecast_domain]["bbox"]
        search_da = spatial_subset(ds[varname], search_bbox)
        if raw_ds is not None:
            raw_da = spatial_subset(raw_ds[varname], forecast_bbox)
        else:
            raw_da = spatial_subset(ds[varname], forecast_bbox)
            
        # set the spatial domain attribute of each, used later in compiling results
        search_da.attrs["spatial_domain"] = search_domain
        raw_da.attrs["spatial_domain"] = forecast_domain
            
        tmp_result, tmp_dates = profile_analog_forecast(search_da, ref_dates, raw_da)
        # profile the analog forecast by computing for all dates
        analog_results.append(tmp_result)
        dates_results.append(tmp_dates)
        
        # profile naive results as well, since we have the data loaded already
        # only need to do this one time as naive results are expected to be the same whether using anomalies or not
        if not use_anom:
            naive_results.append(profile_naive_forecast(raw_da, ref_dates))
        
    analog_df = pd.concat(analog_results)
    analog_df.round(3).to_csv(results_fp, index=False)
    
    dates_df = pd.concat(dates_results)
    dates_fp = str(results_fp).replace(".csv", "_dates.csv")
    dates_df.round(3).to_csv(dates_fp, index=False)
    
    if not use_anom:
        naive_df = pd.concat(naive_results)
        naive_fp = results_fp.parent.joinpath(results_fp.name.replace(".csv", "_naive.csv"))
        naive_df.round(3).to_csv(naive_fp, index=False)
