NAME_TO_VAR = {
    "mean_temperature": "tg",
    "minimum_temperature": "tn",
    "maximum_temperature": "tx",
    "wind_speed_mean": "fg",
    "precipitation_sum": "rr",
    "sea_level_pressure_avg": "pp",
    "relative_humidity_avg": "hu",
    "global_radiation_mean": "qq",
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
}


CONSTANTS = ["orography", "land_sea_mask", "slt", "latitude", "longitude"]

NAME_LEVEL_TO_VAR_LEVEL = {}
for var in SINGLE_LEVEL_VARS:
    NAME_LEVEL_TO_VAR_LEVEL[var] = NAME_TO_VAR[var]

