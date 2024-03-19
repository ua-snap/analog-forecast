"""Module for assisting with ERA5 download scripts"""

import logging
import cdsapi
import pandas as pd
    

def download(download_dir, dataset, varnames, cretrieve_kwargs, fn_suffix):
    """Download ERA5 data via the CDS API"""
    # trying to see if this can be done all in one request (might be too many "elements")
    logging.info(f"Downloading hourly ERA5 data to {download_dir}")
    c = cdsapi.Client()
    
    if isinstance(varnames, str):
        varnames = [varnames]
    
    out_paths = []
    for varname in varnames:
        out_fp = download_dir.joinpath(f"era5_{varname}_{fn_suffix}.nc")
        cretrieve_kwargs.update({"variable": varname})
        c.retrieve(
            dataset,
            cretrieve_kwargs,
            out_fp,
        )
        out_paths.append(out_fp)
    
    return out_paths


def run_download(bbox, ref_date, data_dir, era5_dataset_name, era5_long_name):
    """Get a bounding box for downloading from the CDS API using the luts domain keyword
    
    Args:
        bbox (tuple): tuple of (min lon, min lat, max lon, max lat)
        ref_date (str): string representation of reference date in form YYYY-mm-dd
        data_dir (path-like): path to directory for downloading
        era5_dataset_name (str): name of ERA5 dataset for CDS API
        era5_long_name (str): name of ERA5 variable for CDS API
    """
    # ERA5 at CDS API needs [N, W, S, E]
    bbox = [bbox[-1]] + bbox[:3] 

    ref_date = pd.to_datetime(ref_date)
    cretrieve_kwargs = {   
        "product_type": "reanalysis",
        "format": "netcdf",
        # only going through 2022 right now to avoid getting "expver=5" (initial release data)
        "year": ref_date.year,
        "month": str(ref_date.month).zfill(2),
        "day": str(ref_date.day).zfill(2),
        "time": "12:00",
        "area": bbox,
    }

    out_paths = download(
        data_dir,
        dataset=era5_dataset_name,
        varnames=era5_long_name,
        cretrieve_kwargs=cretrieve_kwargs,
        fn_suffix=f"{ref_date.strftime('%Y%m%d')}"
    )
    
    return out_paths
    