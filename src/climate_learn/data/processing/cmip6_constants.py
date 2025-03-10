NAME_TO_VAR = {
    "surface_pressure": "ps",
    "u_component_of_wind": "uas",
    "v_component_of_wind": "vas",
    "temperature": "ta",
    "specific_humidity": "huss",
    "air_temperature": "tas",
    "precipitation": "pr",
    "geopotential_at_surface": "geopotential_at_surface",
    "latitude": "lat_grid",
    "land_sea_mask": "land_sea_mask",
    "cloud_cover": "clt",
    "upward_heat_flux": "hfls",
    "shortwave_radiation_up": "rsus",
    "shortwave_radiation_down": "rsds",
    "moisture_in_soil": "mrsos",
    "standard_deviation_of_orography": "standard_deviation_of_orography",
    "standard_deviation_of_filtered_subgrid_orography": "standard_deviation_of_filtered_subgrid_orography" ,
    "angle_of_sub_gridscale_orography": "angle_of_sub_gridscale_orography",
    "soil_type": "soil_type"
    
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

SINGLE_LEVEL_VARS = [
    "air_temperature",
    "precipitation",
    "u_component_of_wind",
    "v_component_of_wind",
    "surface_pressure",
    "specific_humidity",
    "geopotential_at_surface",
    "latitude",
    "land_sea_mask",
    "cloud_cover",
    "upward_heat_flux",
    "shortwave_radiation_up",
    "shortwave_radiation_down",
    "moisture_in_soil",
    "standard_deviation_of_orography",
    "standard_deviation_of_filtered_subgrid_orography",
    "angle_of_sub_gridscale_orography",
    "soil_type"
]

PRESSURE_LEVEL_VARS = [
    "temperature", 
]

VAR_TO_UNIT = {
    "air_temperature": "K",
    "precipitation": "mm",
    "geopotential": "m^2/s^2",
    "u_component_of_wind": "m/s",
    "v_component_of_wind": "m/s",
    "temperature": "K",
    "surface_pressure": "Pa",
    "specific_humidity": "kg/kg",
    "geopotential_at_surface": "m",
    "standard_deviation_of_orography": "m",
    "standard_deviation_of_filtered_subgrid_orography": "m",
    "angle_of_sub_gridscale_orography": None,
    "soil_type": None,
    "land_sea_mask": None,  # dimensionless
    "cloud_cover": None,  # dimensionless
    "upward_heat_flux": "W/m^2",
    "up_shortwave_radiation": "W/m^2",
    "shortwave_radiation_up": "W/m^2",
    "shortwave_radiation_down": "W/m^2",
    "moisture_in_soil": "kg/m^2"
}

DEFAULT_PRESSURE_LEVELS = [50, 250, 500, 600, 700, 850, 925]

CONSTANTS = ["geopotential_at_surface", "land_sea_mask", "slt", "lat_grid", "longitude", "standard_deviation_of_orography", "standard_deviation_of_filtered_subgrid_orography", "soil_type", "angle_of_sub_gridscale_orography"]

NAME_LEVEL_TO_VAR_LEVEL = {}

for var in SINGLE_LEVEL_VARS:
    NAME_LEVEL_TO_VAR_LEVEL[var] = NAME_TO_VAR[var]

for var in PRESSURE_LEVEL_VARS:
    for l in DEFAULT_PRESSURE_LEVELS:
        NAME_LEVEL_TO_VAR_LEVEL[var + "_" + str(l)] = NAME_TO_VAR[var] + "_" + str(l)

VAR_LEVEL_TO_NAME_LEVEL = {v: k for k, v in NAME_LEVEL_TO_VAR_LEVEL.items()}
