"""Lookup tables for analog forecasts"""

varnames_lu = {
    "t2m": {
        "filename": "era5_2m_temperature_hour12_1959_2021.nc",
        "anom_filename": "era5_2m_temperature_anom_1959_2021.nc",
    },
    "msl": {
        "filename": "era5_mean_sea_level_pressure_hour12_1959_2021.nc",
        "anom_filename": "era5_mean_sea_level_pressure_anom_1959_2021.nc",
    },
    "sst": {
        "filename": "era5_sea_surface_temperature_hour12_1959_2021.nc",
        "anom_filename": "era5_sea_surface_temperature_anom_1959_2021.nc",
    },
    "z": {
        "filename": "era5_geopotential_hour12_1959_2021.nc",
        "anom_filename": "era5_geopotential_anom_1959_2021.nc",
    }
}

spatial_domains = {
    # min lon, min lat, max lon, max lat
    "alaska": {"bbox": (-180, 44, -125, 76)},
    "northern_hs": {"bbox": (-180, 0, 180, 90)},
    "panarctic": {"bbox": (-180, 55, 180, 90)},
}





