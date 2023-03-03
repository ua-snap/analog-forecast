"""Module for assisting with ERA5 download scripts"""

import logging
import cdsapi


def download(download_dir, dataset, varnames, cretrieve_kwargs, fn_suffix):
    """Download ERA5 data"""
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
    