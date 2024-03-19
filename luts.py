"""Lookup tables for analog forecasts"""

varnames_lu = {
    "t2m": {
        "filename": "era5_2m_temperature_hour12_1959_2021.nc",
        "anom_filename": "era5_2m_temperature_anom_1959_2021.nc",
        "era5_long_name": "2m_temperature",
        "era5_dataset_name": "reanalysis-era5-single-levels",
    },
    "msl": {
        "filename": "era5_mean_sea_level_pressure_hour12_1959_2021.nc",
        "anom_filename": "era5_mean_sea_level_pressure_anom_1959_2021.nc",
        "era5_long_name": "mean_sea_level_pressure",
        "era5_dataset_name": "reanalysis-era5-single-levels",
    },
    "sst": {
        "filename": "era5_sea_surface_temperature_hour12_1959_2021.nc",
        "anom_filename": "era5_sea_surface_temperature_anom_1959_2021.nc",
        "era5_long_name": "sea_surface_temperature",
        "era5_dataset_name": "reanalysis-era5-single-levels",
    },
    "z": {
        "filename": "era5_geopotential_hour12_1959_2021.nc",
        "anom_filename": "era5_geopotential_anom_1959_2021.nc",
        "era5_long_name": "geopotential",
        "era5_dataset_name": "reanalysis-era5-pressure-levels",
    }
}

spatial_domains = {
    # min lon, min lat, max lon, max lat
    "alaska": {"bbox": (-180, 44, -125, 76)},
    "northern_hs": {"bbox": (-180, 0, 180, 90)},
    "panarctic": {"bbox": (-180, 55, 180, 90)},
    "north_pacific": {"bbox": (120, 0, 240, 90)}
}
