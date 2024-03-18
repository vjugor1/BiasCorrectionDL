import numpy as np 
import pandas as pd
import polars
import os, shutil
import sys
import xarray as xr
sys.path.append(os.path.join(os.getcwd()))
from src.regression import data_load as dl

import datetime
from omegaconf import OmegaConf

def str_to_date(string):
    return datetime.datetime.strptime(string, '%Y-%m-%d').date()

def dummy_climate(start_date='2020-12-02', number_of_days=60, lat_lims=(0, 12), lon_lims=(0, 11), dlat=1, dlon=1.15):
    # create dummy data
    climate_vars = ['var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6' ]
    start_date = str_to_date(start_date)
    time_coords = np.array([start_date + datetime.timedelta(days=x) for x in range(number_of_days)]).astype('datetime64')
    lat_coords = np.arange(lat_lims[0], lat_lims[1], dlat)[1:]
    lat_size = len(lat_coords)
    lon_coords = np.arange(lon_lims[0], lon_lims[1], dlon)
    lon_size = len(lon_coords)
    var_1 = np.arange(0, lat_size * lon_size, 1).reshape(lat_size, lon_size) + 0.1
    var_2 = np.arange(0, lat_size * lon_size, 1).reshape(lat_size, lon_size) + 0.2
    var_3 = np.arange(0, lat_size * lon_size, 1).reshape(lat_size, lon_size) + 0.3
    var_4 = np.arange(0, lat_size * lon_size, 1).reshape(lat_size, lon_size) + 0.4
    var_5 = np.arange(0, lat_size * lon_size, 1).reshape(lat_size, lon_size) + 0.5
    var_6 = np.arange(0, lat_size * lon_size, 1).reshape(lat_size, lon_size) + 0.6

    time_slice = np.stack([var_1, var_2, var_3, var_4, var_5, var_6], axis=0)
    data = np.stack([time_slice + 0.001*(i+1) for i in range(number_of_days)], axis=0)

    
    data_xr = xr.DataArray(data, 
    coords={'lat': lat_coords,'lon': lon_coords, 'channel': [0, 1, 2, 3, 4, 5], 'time': time_coords}, 
    dims=["time", "channel", "lat", "lon"])

    assert data.shape[0] == len(time_coords)
    assert data.shape[1] == len(climate_vars)
    assert data.shape[2] == len(lat_coords)
    assert data.shape[3] == len(lon_coords)

    print(f"climate data shape: {data.shape}")
    print(f"time coords shape: {len(time_coords)}")
    print(f"lat coords: {lat_coords}")
    print(f"lon coords: {lon_coords}")

    #dummy config
    cfg = OmegaConf.create({"data_dir": "data/dummy_test", "path_to_prepared_target_data": "data/dummy_test/target.parquet",
    "precision": 16, "normalize": False, "time_window": 3, "half_side_size": 3, "start_of_test": str(time_coords[number_of_days // 2]), "variables": climate_vars})
    if not os.path.exists(cfg.train.data_dir):
        os.makedirs(cfg.train.data_dir)
    
    # saving test files
    with open(os.path.join(cfg.train.data_dir, 'lat.npy'), 'wb') as f:
        np.save(f, lat_coords)
    with open(os.path.join(cfg.train.data_dir, 'lon.npy'), 'wb') as f:
        np.save(f, lon_coords)
    with open(os.path.join(cfg.train.data_dir, 'time.npy'), 'wb') as f:
        np.save(f, time_coords)
        pass
    dummy_mean = np.zeros(len(climate_vars))
    with open(os.path.join(cfg.train.data_dir, 'mean_16.npy'), 'wb') as f:
        np.save(f, dummy_mean)
    with open(os.path.join(cfg.train.data_dir, 'mean_32.npy'), 'wb') as f:
        np.save(f, dummy_mean)
    dummy_std = np.ones(len(climate_vars))
    with open(os.path.join(cfg.train.data_dir, 'std_16.npy'), 'wb') as f:
        np.save(f, dummy_std)
    with open(os.path.join(cfg.train.data_dir, 'std_32.npy'), 'wb') as f:
        np.save(f, dummy_std)
    for i, name in enumerate(climate_vars):
        with open(os.path.join(cfg.train.data_dir, name + f'_{cfg.process.precision}' + '.npy'), 'wb') as f:
            np.save(f, data[:, i, :, :])

    # df_data = {"time": np.concatenate((time_coords[0:6],time_coords[0:6])),
    #            "y": [3, 7] * 3 + [3, 7] * 3,
    #            "lat": [3, 7] * 3 + [3, 7] * 3,
    #            "lon": [3, 7] * 3 + [3, 7] * 3,
    #         #    "station_name": ["A"] * 6 + ["B"] * 6
    #           }
    df_data_ru = {
        "time": np.concatenate((time_coords, time_coords, time_coords)),
        "y": np.concatenate((np.arange(len(time_coords)), np.arange(len(time_coords)), np.arange(len(time_coords)))),
        "station_name": ["A"] * len(time_coords) + ["B"] * len(time_coords) + ["C"] * len(time_coords),
                }
    df_data_ru = polars.from_dict(df_data_ru)
    stations_df = {
        "station_name": ["A", "B", "C"],
        "lat": [0, 3.65, 3.7],
        "lon": [0, 3.95, 3.7],
        }
    stations_df = polars.from_dict(stations_df)
    
    # df_data_ru.to_parquet(os.path.join(cfg.train.data_dir, 'target.parquet'))

    return data, data_xr, time_coords, lat_coords, lon_coords, climate_vars, df_data_ru, stations_df, cfg

if __name__ == '__main__':
    dummy_climate()