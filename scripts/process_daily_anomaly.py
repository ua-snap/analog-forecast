"""Compute the daily anomalies for the downloaded ERA5 data"""

import argparse
import xarray as xr
from config import data_dir
from dask.distributed import Client
import luts
from dask.diagnostics import ProgressBar


def parse_args():
    """Parse some arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v",
        dest="varname",
        type=str,
        help=(
            "Variable name to generate daily anomaly dataset for"
            "Options are t2m (temperature), sst (sea surface temperature), msl (mean sea level pressure), z (geopotential 500 hPa)"
        ),
        required=True
    )
    args = parser.parse_args()
    
    return args.varname


def calculate_anomaly(da):
    """Function to calculate anomaly for a dataarray, to be called by map_blocks"""
    gb = da.groupby("time.dayofyear")
    clim = gb.mean(dim="time")
    return gb - clim


if __name__ == "__main__":
    varname = parse_args()
    
    # open connection to file
    fp = data_dir.joinpath(luts.varnames_lu[varname]["filename"])
    ds = xr.open_dataset(fp)

    # Probably a better way to do this but we just need to tell map_blocks what the resulting dataarray will look like,
    #  which is the same as the original dataarray but with the addition of a dayofyear coordinate variable indexed
    #  by the time dimension
    da = xr.DataArray(
        ds[varname].data,
        dims=["time", "latitude", "longitude"],
        coords={
            "time": ds.time,
            "dayofyear": ds.time.dt.dayofyear,
            "latitude": ds.latitude,
            "longitude": ds.longitude
        },
    )

    daily_anom = da.map_blocks(calculate_anomaly, template=da)
    daily_anom.name = varname

    delayed_write = daily_anom.to_dataset().to_netcdf(
        data_dir.joinpath(luts.varnames_lu[varname]["anom_filename"]),
        compute=False
    )
    
    with ProgressBar():
        results = delayed_write.compute()
