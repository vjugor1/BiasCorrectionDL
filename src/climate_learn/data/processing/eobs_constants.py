NAME_TO_VAR = {
    "mean_temperature": "tg",
    "minimum_temperature": "tn",
    "maximum_temperature": "tx",
    "wind_speed_mean": "fg",
    "rainfall": "rr"
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

SINGLE_LEVEL_VARS = [
    "mean_temperature",
    "minimum_temperature",
    "maximum_temperature",
    "wind_speed_mean",
    "rainfall"
]


VAR_TO_UNIT = {
    "mean_temperature": "C",
    "minimum_temperature": "C",
    "maximum_temperature": "C",
    "wind_speed_mean": "m/s",
    "rainfall": "mm"
}

CONSTANTS = ["orography", "land_sea_mask", "slt", "lattitude", "longitude"]

NAME_LEVEL_TO_VAR_LEVEL = {}
for var in SINGLE_LEVEL_VARS:
    NAME_LEVEL_TO_VAR_LEVEL[var] = NAME_TO_VAR[var]

