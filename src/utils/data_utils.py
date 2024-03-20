import numpy as np
from scipy import interpolate
import xarray as xr

from calendar import isleap
import torch
# import pandas as pd


def round_to_closest_indices(arr, values):
    values = np.array(values)
    indices = np.searchsorted(values, arr)
    indices = np.clip(indices, 1, len(values) - 1)
    left_values = values[indices - 1]
    right_values = values[indices]
    left_indices = indices - 1
    right_indices = indices
    closest_indices = np.where(np.abs(arr - left_values) <= np.abs(arr - right_values), left_indices, right_indices)
    return closest_indices


def round_to_closest_values(arr, values):
    values = np.array(values)
    indices = np.searchsorted(values, arr)
    indices = np.clip(indices, 1, len(values) - 1)
    left_values = values[indices - 1]
    right_values = values[indices]
    closest_values = np.where(np.abs(arr - left_values) <= np.abs(arr - right_values), left_values, right_values)
    return closest_values


def make_padding(data, half_side_size):
    quadrants, fourth_q_shape = extract_quadrants(data)
    quadrants_borders = extrect_quadrant_borders(quadrants, half_side_size)
    padded_map = assemble_padded_map(quadrants, quadrants_borders, half_side_size)
    return padded_map, (half_side_size, half_side_size)


def extract_quadrants(data):
    halfs = {"lat": data.shape[-2] // 2, "lon": data.shape[-1] // 2}
    first_quadrant  = data[..., halfs['lat']:, halfs['lon']:]
    second_quadrant = data[..., halfs['lat']:, :halfs['lon']]
    third_quadrant  = data[..., :halfs['lat'], :halfs['lon']]
    fourth_quadrant = data[..., :halfs['lat'], halfs['lon']:]
    fourth_q_shape = (fourth_quadrant.shape[-2], fourth_quadrant.shape[-1])
    return (first_quadrant, second_quadrant, third_quadrant, fourth_quadrant), fourth_q_shape
    

def extrect_quadrant_borders(quadrants, half_side_size):
    q_borders = {}

    q_borders['0_top'] = quadrants[0][..., -half_side_size:, :]
    q_borders['0_right'] = quadrants[0][..., :, -half_side_size:]

    q_borders['1_top'] = quadrants[1][..., -half_side_size:, :]
    q_borders['1_left'] = quadrants[1][..., :, :half_side_size]

    q_borders['2_bot'] = quadrants[2][..., :half_side_size, :]
    q_borders['2_left'] = quadrants[2][..., :, :half_side_size]

    q_borders['3_bot'] = quadrants[3][..., :half_side_size, :]
    q_borders['3_right'] = quadrants[3][..., :, -half_side_size:]

    return q_borders


def assemble_padded_map(quadrants, q_borders, half_side_size):
    try:
        column_1 = np.concatenate((q_borders['3_bot'].reindex(lat=list(reversed(q_borders['3_bot'].lat))), quadrants[2], quadrants[1], q_borders['0_top'].reindex(lat=list(reversed(q_borders['0_top'].lat)))), axis=-2)
        column_2 = np.concatenate((q_borders['2_bot'].reindex(lat=list(reversed(q_borders['2_bot'].lat))), quadrants[3], quadrants[0], q_borders['1_top'].reindex(lat=list(reversed(q_borders['1_top'].lat)))), axis=-2)
    except AttributeError:
        column_1 = np.concatenate((np.flip(q_borders['3_bot'], axis=-2), quadrants[2], quadrants[1], np.flip(q_borders['0_top'], axis=-2)), axis=-2)
        column_2 = np.concatenate((np.flip(q_borders['2_bot'], axis=-2), quadrants[3], quadrants[0], np.flip(q_borders['1_top'], axis=-2)), axis=-2)
    column_0 = column_2[..., :, -half_side_size:]
    column_3 = column_1[..., :, :half_side_size]

    padded_map = np.concatenate((column_0, column_1, column_2, column_3), axis=-1)
    return padded_map

def make_padding_torch(data, half_side_size):
    quadrants, fourth_q_shape = extract_quadrants(data)
    quadrants_borders = extrect_quadrant_borders(quadrants, half_side_size)
    padded_map = assemble_padded_map_torch(quadrants, quadrants_borders, half_side_size)
    return padded_map, (half_side_size, half_side_size)



def assemble_padded_map_torch(quadrants, q_borders, half_side_size):
    try:
        column_1 = torch.concatenate((q_borders['3_bot'].reindex(lat=list(reversed(q_borders['3_bot'].lat))), quadrants[2], quadrants[1], q_borders['0_top'].reindex(lat=list(reversed(q_borders['0_top'].lat)))), dims=(-2,))
        column_2 = torch.concatenate((q_borders['2_bot'].reindex(lat=list(reversed(q_borders['2_bot'].lat))), quadrants[3], quadrants[0], q_borders['1_top'].reindex(lat=list(reversed(q_borders['1_top'].lat)))), dims=(-2,))
    except AttributeError:
        column_1 = torch.concatenate((torch.flip(q_borders['3_bot'], dims=(-2,)), quadrants[2], quadrants[1], torch.flip(q_borders['0_top'], dims=(-2,))), dim=-2)
        column_2 = torch.concatenate((torch.flip(q_borders['2_bot'], dims=(-2,)), quadrants[3], quadrants[0], torch.flip(q_borders['1_top'], dims=(-2,))), dim=-2)
    column_0 = column_2[..., :, -half_side_size:]
    column_3 = column_1[..., :, :half_side_size]

    padded_map = torch.concatenate((column_0, column_1, column_2, column_3), axis=-1)
    return padded_map


NAME_TO_VAR = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
    "surface_pressure": "sp",
    "toa_incident_solar_radiation": "tisr",
    "total_precipitation": "tp",
    "land_sea_mask": "lsm",
    "orography": "orography",
    "lattitude": "lat2d",
    "geopotential": "z",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "temperature": "t",
    "relative_humidity": "r",
    "specific_humidity": "q",
}

VAR_TO_NAME = {v: k for k, v in NAME_TO_VAR.items()}

SINGLE_LEVEL_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "surface_pressure",
    "toa_incident_solar_radiation",
    "total_precipitation",
    "land_sea_mask",
    "orography",
    "lattitude",
]
PRESSURE_LEVEL_VARS = [
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "relative_humidity",
    "specific_humidity",
]
DEFAULT_PRESSURE_LEVELS = [50, 250, 500, 600, 700, 850, 925, 1000]

NAME_LEVEL_TO_VAR_LEVEL = {}

for var in SINGLE_LEVEL_VARS:
    NAME_LEVEL_TO_VAR_LEVEL[var] = NAME_TO_VAR[var]

for var in PRESSURE_LEVEL_VARS:
    for l in DEFAULT_PRESSURE_LEVELS:
        NAME_LEVEL_TO_VAR_LEVEL[var + "_" + str(l)] = NAME_TO_VAR[var] + "_" + str(l)

VAR_LEVEL_TO_NAME_LEVEL = {v: k for k, v in NAME_LEVEL_TO_VAR_LEVEL.items()}

BOUNDARIES = {
    'NorthAmerica': { # 8x14
        'lat_range': (15, 65),
        'lon_range': (220, 300)
    },
    'SouthAmerica': { # 14x10
        'lat_range': (-55, 20),
        'lon_range': (270, 330)
    },
    'Europe': { # 6x8
        'lat_range': (30, 65),
        'lon_range': (0, 40)
    },
    'SouthAsia': { # 10, 14
        'lat_range': (-15, 45),
        'lon_range': (25, 110)
    },
    'EastAsia': { # 10, 12
        'lat_range': (5, 65),
        'lon_range': (70, 150)
    },
    'Australia': { # 10x14
        'lat_range': (-50, 10),
        'lon_range': (100, 180)
    },
    'Global': { # 32, 64
        'lat_range': (-90, 90),
        'lon_range': (0, 360)
    }
}

def interp_timewise_xarray(data_arr: xr.DataArray,
                            res: float = 0.25,
                            interp_method: str = 'linear') -> xr.DataArray:
    """ Interpolate a xarray DataArray in 3d """
    
    y = data_arr.lon.data
    x = data_arr.lat.data
    ynew = np.arange(y[0], y[-1], 1)
    xnew = np.arange(x[0], x[-1], 1)
    timesteps = (data_arr.time.data - np.datetime64('1970-01-01T00:00:00Z'))/ np.timedelta64(1, 's')
    sample_at_t, sample_at_x, sample_at_y  = np.meshgrid(timesteps, xnew, ynew, indexing='ij') 
    points = (timesteps, x, y)

    interp_data = interpolate.interpn(points=points,
                                      values=data_arr.data.compute(),
                                      xi=(sample_at_t, sample_at_x, sample_at_y),
                                      method='linear')        

    coords_new = {'time': ("time", data_arr.time.data), 'lat': ("lat", xnew), 'lon': ("lon", ynew)}
    output = xr.DataArray(data=interp_data, coords=coords_new, dims=('time', 'lat', 'lon'), attrs=data_arr.attrs)

    return output


def get_region_info(region, lat, lon, patch_size):
    region = BOUNDARIES[region]
    lat_range = region['lat_range']
    lon_range = region['lon_range']
    lat = lat[::-1] # -90 to 90 from south (bottom) to north (top)
    h, w = len(lat), len(lon)
    lat_matrix = np.expand_dims(lat, axis=1).repeat(w, axis=1)
    lon_matrix = np.expand_dims(lon, axis=0).repeat(h, axis=0)
    valid_cells = (lat_matrix >= lat_range[0]) & (lat_matrix <= lat_range[1]) & (lon_matrix >= lon_range[0]) & (lon_matrix <= lon_range[1])
    h_ids, w_ids = np.nonzero(valid_cells)
    h_from, h_to = h_ids[0], h_ids[-1]
    w_from, w_to = w_ids[0], w_ids[-1]
    patch_idx = -1
    p = patch_size
    valid_patch_ids = []
    min_h, max_h = 1e5, -1e5
    min_w, max_w = 1e5, -1e5
    for i in range(0, h, p):
        for j in range(0, w, p):
            patch_idx += 1
            if (i >= h_from) & (i + p - 1 <= h_to) & (j >= w_from) & (j + p - 1 <= w_to):
                valid_patch_ids.append(patch_idx)
                min_h = min(min_h, i)
                max_h = max(max_h, i + p - 1)
                min_w = min(min_w, j)
                max_w = max(max_w, j + p - 1)
    return {
        'patch_ids': valid_patch_ids,
        'min_h': min_h,
        'max_h': max_h,
        'min_w': min_w,
        'max_w': max_w
    }

def cleanup_ms_name(name: str):
    name = name.replace('"', '')
    name = name.replace(',', '')
    return name


def check_leap_year(date):
    year = date.astype('datetime64[Y]').astype(int) + 1970

    return np.logical_not(
        np.logical_and(np.not_equal(year % 4, 0), np.logical_or(np.not_equal(year % 100, 0), np.equal(year % 400, 0))))


def test_check_leap_year(tmp: xr.DataArray):
    t0 = np.apply_along_axis(check_leap_year, axis=0, arr=tmp.time.data)
    t1 = np.array([isleap(v) for v in (tmp.time.data.astype('datetime64[Y]').astype(int) + 1970)])
    assert (t1 == t0).all(), "check_leap_year does not work correctly"


def read_splits(train_path, val_path, test_path):
    with open(train_path) as f:
        train_list = f.read().split('\n')
    with open(test_path) as f:
        test_list = f.read().split('\n')
    if val_path:
        with open(test_path) as f:
            val_list = f.read().split('\n')
    else:
        val_list = test_list
    return train_list, val_list, test_list




# df = pd.read_csv("./data/weather_stations/data_meteo_full.csv", engine="pyarrow")
# df.to_parquet("./data/weather_stations/data_meteo_full.parquet", compression=None)
