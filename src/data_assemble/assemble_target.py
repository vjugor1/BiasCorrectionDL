import sys,os
sys.path.append(os.getcwd())
import pandas as pd
from functools import partial
import numpy as np
import logging
import xarray as xr
import time
from omegaconf import DictConfig, ListConfig
import gc
import polars as pl


def preprocess_coordinates(): # TODO 
    pass


def cleanup_ms_name(name: str):        
        name = name.replace('"', '')
        name = name.replace(',', '')
        name = name.casefold()
        return name

def get_stations_RU(cfg) -> pd.DataFrame:

    all_stations_data = cfg.raw.path_to_weather_station_list
    rectangle_coords = list(cfg.process.coords.values())                                   
    min_lat=rectangle_coords[0]
    max_lat=rectangle_coords[1]                                    
    min_lon=rectangle_coords[2]
    max_lon=rectangle_coords[3]
    max_height = cfg.process.max_height
    min_height = cfg.process.min_height
    all_stations = pd.read_json(all_stations_data)
    all_stations = all_stations.rename(columns={
                            "Широта": "lat",
                            "Долгота": "lon",
                            "Наименование станции": "station_name",
                            "Высота метеопл.": "height"
                            })    
    all_stations["station_name"] = all_stations["station_name"].apply(cleanup_ms_name)
    all_stations["lat"] = all_stations["lat"].astype(np.float32)
    all_stations["lon"] = all_stations["lon"].astype(np.float32)
    all_stations["height"] = all_stations["height"].astype(np.float32)
    result_stations = all_stations[["station_name", "lat", "lon", "height"]]
    if cfg.process.spatial_crop:
        if max_lat:
            result_stations = result_stations[result_stations['lat'] < max_lat]
        if min_lat:
            result_stations = result_stations[result_stations['lat'] > min_lat]
        if max_lon:
            result_stations = result_stations[result_stations['lon'] < max_lon]
        if min_lon:
            result_stations = result_stations[result_stations['lon'] > min_lon]
        if max_height:
            result_stations = result_stations[result_stations['height'] < max_height]
        if min_height:
            result_stations = result_stations[result_stations['height'] > min_height]
    
    return result_stations



def clean_weather_data_RU(path_to_weather_stations: str) -> pd.DataFrame:
    """ 
    To load weather stations data from Russia 
        Features: ['Максимальная скорость ветра', 'Средняя скорость ветра', 'Направление ветра', 
                   'Температура воздуха по сухому терм-ру', 'Атмосферное давление на уровне станции', 
                   'Атмосферное давление на уровне моря', 'Сумма осадков', 'Температура поверхности почвы',
                   'Парциальное давление водяного пара', 'Относительная влажность воздуха', 'Температура точки росы'] 
    """  
    columns = ["Название метеостанции",
               "Максимальная скорость",
               'Средняя скорость ветра',
               'Температура воздуха по сухому терм-ру',
               'Температура точки росы',
               'Атмосферное давление на уровне станции',
               'Атмосферное давление на уровне моря',
               'Сумма осадков',
               "Дата"] # Add more columns if needed
    
    start_time = time.process_time()
    df = pl.read_parquet(path_to_weather_stations, columns=columns)
    logging.info(f"Time to open ru parquet {time.process_time() - start_time} seconds")
    df = (df
        .lazy()
        .select(
            [
                pl.col("Название метеостанции").apply(cleanup_ms_name).cast(pl.Categorical).alias("station_name"),
                pl.col("Максимальная скорость").round().cast(pl.Float32).alias("max_speed"),
                pl.col("Средняя скорость ветра").round().cast(pl.Float32).alias("avg_speed"),
                pl.col("Дата").cast(pl.Datetime).alias("time"),
                pl.col("Дата").cast(pl.Date).alias("date"),
                pl.col("Температура воздуха по сухому терм-ру").round().cast(pl.Float32).alias("avg_temp"),
                pl.col("Температура точки росы").round().cast(pl.Float32).alias("dew_point_temp"),
                pl.col("Сумма осадков").round().cast(pl.Float32).alias("precipitation"),
                pl.col("Атмосферное давление на уровне станции").round().cast(pl.Float32).alias("station_level_pressure"),
                pl.col("Атмосферное давление на уровне моря").round().cast(pl.Float32).alias("sea_level_pressure"),
            ]
               )
        )
    df = df.collect()
    df = (df
        .lazy()
        .groupby([pl.col("station_name"), pl.col("date")])
        .agg([pl.col("max_speed").max(),
              pl.col("avg_speed").mean(),
              pl.col("avg_temp").mean(),
              pl.col("dew_point_temp").mean(),
              pl.col("station_level_pressure").mean(),
              pl.col("sea_level_pressure").mean()])
        )
    df = df.collect()
    df = df.rename({'date': 'time'})
    df.write_parquet(path_to_weather_stations.replace(".parquet", "_cleaned.parquet"))



def clean_weather_data_WORLD(path_to_weather_stations: str) -> pd.DataFrame:
    """ To load weather stations data from all world 
        Features: ['DATE', 'STATION', 'NAME', 'MXWDSP', 'WDSP', 'TEMP', 'STP', 'SLP',
       'PRCP', 'DEWP', 'LATITUDE', 'LONGITUDE', 'ELEVATION'] 
    """  

    columns = ["STATION", "LATITUDE",  "LONGITUDE", "ELEVATION", "DATE", 'MXWDSP', 'WDSP', 'TEMP', 'DEWP', 'SLP', 'STP', 'PRCP'] 
    start_time = time.process_time()
    df = pl.read_parquet(path_to_weather_stations, columns=columns)
    logging.info(f"Time to open world parquet {time.process_time() - start_time} seconds")
    q = (df
        .lazy()
        .select(
            [
                pl.col("STATION").cast(pl.Categorical).alias("station_name"),
                pl.col("MXWDSP").round().cast(pl.Float32).alias("max_speed"),
                pl.col("WDSP").round().cast(pl.Float32).alias("avg_speed"),
                pl.col("DATE").cast(pl.Date).alias("time"),                
                pl.col("TEMP").round().cast(pl.Float32).alias("avg_temp"),
                pl.col("DEWP").round().cast(pl.Float32).alias("dew_point_temp"),
                pl.col("STP").round().cast(pl.Float32).alias("station_level_pressure"),
                pl.col("SLP").round().cast(pl.Float32).alias("sea_level_pressure"),
                pl.col("PRCP").round().cast(pl.Float32).alias("precipitation"),
                pl.col("LATITUDE").round().cast(pl.Float32).alias("lat"),
                pl.col("LONGITUDE").round().cast(pl.Float32).alias("lon"),
                pl.col("ELEVATION").round().cast(pl.Float32).alias("height"),
            ]
               )
        )
    
    q = q.collect()
    q.write_parquet(path_to_weather_stations.replace(".parquet", "_cleaned.parquet"))    


def filter_lat_lon(stations_df, cfg):
    lat_min, lat_max, lon_min, lon_max = cfg.process.coords.lat_min, cfg.process.coords.lat_max, cfg.process.coords.lon_min, cfg.process.coords.lon_max
    return stations_df.filter((pl.col('lat') >= lat_min) & (pl.col('lat') <= lat_max) & (pl.col('lon') <= lon_max) & (pl.col('lon') >= lon_min))


def pre_prepare_target_RU(cfg: DictConfig, dataset_xarray: xr.DataArray):
        
    start_time = time.process_time()
    df = pl.read_parquet(cfg.raw.path_to_weather_stations_data.replace(".parquet", "_cleaned.parquet"), use_pyarrow=True)
    logging.info(f"Time to open ru parquet {time.process_time() - start_time} seconds")
    start = cfg.process.time_limits[0]
    end = cfg.process.time_limits[1]
    df = df.filter(pl.any(pl.col('time') >= pd.to_datetime(start)))
    df = df.filter(pl.any(pl.col('time') <= pd.to_datetime(end)))
    if len(cfg.process.target_column)>1:
        target_cols = [df[col] for col in cfg.process.target_column]
        df['y'] = list(zip(*target_cols)) 
    elif len(cfg.process.target_column)==1:
        df = df.rename({cfg.process.target_column[0]: 'y'})    
    else:
        raise ValueError
    
    df = df.select(pl.col(["time", "station_name", "y" ]))    
    stations_df_ru = pl.from_pandas(get_stations_RU(cfg))
    stations_df_ru = filter_lat_lon(stations_df_ru, cfg)      
    # stations_df_ru = stations_to_data_grid(dataset_xarray=dataset_xarray,
    #                                        stations_df=stations_df_ru)
    df = df.select([pl.all().exclude("station_name"), pl.col("station_name").cast(str).keep_name()])   
    df = df.join(stations_df_ru, on='station_name', how='left')
    df = df.select(pl.col(["time", "y", "lat", "lon"])).drop_nulls()
    print(df)
    df.write_parquet(os.path.join(cfg.process.data_dir, cfg.process.prepared_target_data_name + '.pp1'))

    

def pre_prepare_target_WORLD(cfg: DictConfig, dataset_xarray: xr.DataArray):

    start_time = time.process_time()

    df = pl.read_parquet(cfg.raw.path_to_world_weather_stations_data.replace(".parquet", "_cleaned.parquet"), use_pyarrow=True)
    logging.info(f"Time to open world parquet {time.process_time() - start_time} seconds")
    df = df.drop_nulls()

    start = cfg.process.time_limits[0]
    end = cfg.process.time_limits[1]
    df = df.filter(pl.col('time') >= pd.to_datetime(start))
    df = df.filter(pl.col('time') <= pd.to_datetime(end))
    df = filter_lat_lon(df, cfg)
    if len(cfg.process.target_column)>1:
        target_cols = [df[col] for col in cfg.process.target_column]
        df['y'] = list(zip(*target_cols)) 
    elif len(cfg.process.target_column)==1:
        df = df.rename({cfg.process.target_column[0]: 'y'})    
    else:
        raise ValueError
        
    df = df.select(pl.col(["time", "station_name", "y", "lat", "lon"]))
    gc.collect()      
    # df = stations_to_data_grid(dataset_xarray=dataset_xarray, stations_df=df)
    df = df.drop("station_name")
    df.write_parquet(os.path.join(cfg.process.data_dir, cfg.process.prepared_target_data_name + '.pp2'))



def make_target(cfg: DictConfig, dataset_xarray: xr.DataArray):    

    # pre_prepare_target_RU(cfg, dataset_xarray)
    pre_prepare_target_WORLD(cfg, dataset_xarray)

    # df_ru = pl.read_parquet(os.path.join(cfg.process.data_dir, cfg.process.prepared_target_data_name + '.pp1'), use_pyarrow=True)
    # logging.info(f'RU len: {len(df_ru)}')
    df_world = pl.read_parquet(os.path.join(cfg.process.data_dir, cfg.process.prepared_target_data_name + '.pp2'), use_pyarrow=True)
    logging.info(f'WORLD len: {len(df_world)}')
    start_time = time.process_time()
    # target_df = pl.concat([df_ru, df_world], how='diagonal')
    target_df = df_world
    print(target_df)
    logging.info(f"Concat took {time.process_time() - start_time} seconds")
    target_df = target_df.drop_nulls()
    logging.info(f'TOTAL len: {len(target_df)}')
    target_df.write_parquet(os.path.join(cfg.process.data_dir, cfg.process.prepared_target_data_name))