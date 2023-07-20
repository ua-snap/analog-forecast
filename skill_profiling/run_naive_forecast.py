"""Run a naive forecast for a given variable and forecast domain"""


import argparse
from pathlib import Path
import dask
import numpy as np
import pandas as pd
import xarray as xr
from multiprocessing import Pool
import luts
from analog_forecast import spatial_subset
from skill_profiling.run_profile import forecast_dates, generate_reference_dates, forecast_and_error
from config import data_dir


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--varname",
        type=str,
        help="Variable name",
        required=True
    )
    parser.add_argument(
        "--results_fp",
        type=str,
        help="path to write results to",
        required=True
    )
    args = parser.parse_args()
    
    return args.varname, Path(args.results_fp)


def get_naive_sample_dates(all_times, ref_date, n=1000):
    """Constructs list of all dates to be used for a single naive forecast and error. 
    
    Args:
        all_times (list): list of all times available in the hsitorical archive
        ref_date (str): reference date in format YYYY-mm-dd
        n (int): number of repetitions to select from available dates
        
    Returns:
        naive_dates (dict): dictionary of info for each of n reps, with each element being a dict of analogs and "all dates" associated with those analogs, i.e. a list of the analog dates and the subsequent dates of each to be used in the forecast
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
    
    # choose 5 times at random n times
    naive_dates = {
        f"rep{rep}": {
            "analogs": list(np.random.choice(keep_times, 5, replace=False))
        } for rep in range(n)
    }
    
    # get "all dates" for a given forecast (the 5 analogs plus dates for the forecast) n times
    all_dates = []
    for rep in naive_dates:
        rep_analog_dates = naive_dates[rep]["analogs"]
        all_rep_dates = []
        for t in rep_analog_dates + [ref_dt]:
            all_rep_dates.extend(pd.date_range(t, t + pd.to_timedelta(forecast_length, "D")))
        naive_dates[rep]["all_dates"] = list(pd.Series(sorted(all_rep_dates)).drop_duplicates())

    return naive_dates


@dask.delayed
def run_forecast_and_error(args):
    """dask.delayed wrapper for forecast_and_error"""
    return forecast_and_error(*args)


def profile_naive_forecast(raw_da, ref_dates, n=1000):
    """Profile the naive forecast by simulating it n times.
    
    Args:
        raw_da (xarray.DataArray): 
        ref_dates (list): list of string representations of dates in form YYYY-mm-dd
        n (int): number of simulations to run
        
    Returns:
        err_df (pandas.DataFrame): profiling results for the naive forecast for all reference dates, for the supplied dataArray
    """
    err_df_rows = []
    for ref_date in ref_dates:
        all_naive_dates = get_naive_sample_dates(raw_da.time.values, ref_date, n=n)
        args = [
            (
                raw_da.sel(time=all_naive_dates[rep]["all_dates"]),
                all_naive_dates[rep]["analogs"],
                ref_date
            )
            for rep in all_naive_dates
        ]

        results = []
        for arg in args:
            results.append(run_forecast_and_error(arg))
        
        results = dask.compute(results)
        sim_rmse = xr.concat(results[0], pd.Index(range(n), name="sim"))

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
    varname, results_fp = parse_args()
    ref_dates = generate_reference_dates(forecast_dates)
    raw_fp = data_dir.joinpath(luts.varnames_lu[varname]["filename"])
    raw_ds = xr.open_dataset(raw_fp)
    naive_results = []
    for forecast_domain in luts.spatial_domains:
        forecast_bbox = luts.spatial_domains[forecast_domain]["bbox"]
        raw_da = spatial_subset(raw_ds[varname], forecast_bbox)
        raw_da.attrs["spatial_domain"] = forecast_domain
        results = profile_naive_forecast(raw_da, ref_dates)
        naive_results.append(results)
        print(forecast_domain, "done")
    
    naive_df = pd.concat(naive_results)
    naive_df.round(3).to_csv(results_fp, index=False)
