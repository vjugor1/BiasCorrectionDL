NAME_TO_VAR = {
    "mean_temperature": "tg",
    "minimum_temperature": "tn",
    "maximum_temperature": "tx",
    "wind_speed_mean": "fg",
    "precipitation_sum": "rr",
    "sea_level_pressure_avg": "pp",
    "relative_humidity_avg": "hu",
    "global_radiation_mean": "qq",
    "standard_deviation_of_orography": "standard_deviation_of_orography",
    "standard_deviation_of_filtered_subgrid_orography": "standard_deviation_of_filtered_subgrid_orography" ,
    "angle_of_sub_gridscale_orography": "angle_of_sub_gridscale_orography",
    "soil_type": "soil_type",
    "geopotential_at_surface": "geopotential_at_surface",
    "latitude": "lat_grid",
    "land_sea_mask": "land_sea_mask",
}


VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

SINGLE_LEVEL_VARS = [
    "mean_temperature",
    "minimum_temperature",
    "maximum_temperature",
    "wind_speed_mean",
    "precipitation_sum",
    "sea_level_pressure_avg",
    "relative_humidity_avg",
    "global_radiation_mean",
    "standard_deviation_of_orography",
    "standard_deviation_of_filtered_subgrid_orography",
    "angle_of_sub_gridscale_orography",
    "soil_type",
    "geopotential_at_surface",
    "latitude",
    "land_sea_mask"
    
]


VAR_TO_UNIT = {
    "mean_temperature": "C",
    "minimum_temperature": "C",
    "maximum_temperature": "C",
    "wind_speed_mean": "m/s",
    "precipitation_sum": "mm",
    "sea_level_pressure_avg": "hPa",
    "relative_humidity_avg": "%",
    "global_radiation_mean": "W/m2",
    "tg": "C",
    "tn": "C",
    "tx": "C",
    "rr": "mm",
    "geopotential_at_surface": "m",
    "standard_deviation_of_orography": "m",
    "standard_deviation_of_filtered_subgrid_orography": "m",
    "angle_of_sub_gridscale_orography": None,
    "soil_type": None,
    "land_sea_mask": None,  # dimensionless
}


CONSTANTS = ["orography", "land_sea_mask", "slt", "latitude", "geopotential_at_surface", "lat_grid", "longitude", "standard_deviation_of_orography", "standard_deviation_of_filtered_subgrid_orography", "soil_type", "angle_of_sub_gridscale_orography"]

NAME_LEVEL_TO_VAR_LEVEL = {}
for var in SINGLE_LEVEL_VARS:
    NAME_LEVEL_TO_VAR_LEVEL[var] = NAME_TO_VAR[var]

