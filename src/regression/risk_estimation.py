import sys,os
sys.path.append(os.getcwd())
import logging
import warnings
import random
from omegaconf import DictConfig, OmegaConf
import time
import hydra

import pandas as pd
import xarray as xr
import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
import geopandas as gpd
import fiona
from fiona.drvsupport import supported_drivers
supported_drivers['LIBKML'] = 'rw'

warnings.filterwarnings("ignore")
random.seed(112)

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')

def interpolate(df_risks: pd.DataFrame, cfg: DictConfig) -> None:
    dlat = np.abs(np.diff(df_risks.lat.drop_duplicates())).min()
    dlon = np.abs(np.diff(df_risks.lon.drop_duplicates())).min()
    assert (dlat > cfg.eval.interpolation_res) and (dlon > cfg.eval.interpolation_res), "Current spatial resolution is lower than interpolation target"

    reshaped = df_risks.prob.values.reshape(len(df_risks.lat.unique()), len(df_risks.lon.unique()), len(df_risks.timestamp.unique()))

    data_xr = xr.DataArray(reshaped, 
    coords={'lat': np.sort(df_risks.lat.unique()),'lon': np.sort(df_risks.lon.unique()),'timestamp': np.sort(df_risks.timestamp.unique())}, 
    dims=["lat", "lon", "timestamp"])

    new_lons = np.arange(data_xr.lon.min(), data_xr.lon.max(), cfg.eval.interpolation_res)
    new_lats = np.arange(data_xr.lat.min(), data_xr.lat.max(), cfg.eval.interpolation_res)
    data_xr_interpolated = data_xr.interp(lon=new_lons, lat=new_lats)

    data_xr_interpolated.name = 'prob'
    df = data_xr_interpolated.to_dataframe()
    df = df.reset_index(level=[0,1])
    # df['timestamp'] = df.index
    df.reset_index(inplace=True)
    df_risks = df

    return df_risks


def risk_estimation(cfg: DictConfig) -> None:        
    start_time = time.process_time()  
    logging.info(f"Reading raw inference")
    df_infer = pd.read_csv(cfg.eval.path_to_predictions)

    logging.info(f"Grouping by months, estimating risk")
    logging.info(f"Grouping by months, estimating risk")
    df_infer['date'] = pd.to_datetime(df_infer['date'])
    df_grpby = df_infer.groupby(['lat', 'lon', df_infer.date.dt.year, df_infer.date.dt.month])
    df_risks = df_grpby['prediction'].agg(lambda x: (x > cfg.eval.wind_risk_threshold).mean())
    df_risks_ = df_risks.index.rename(['lat', 'lon', 'year', 'month']).to_frame().reset_index(drop=True)
    df_risks_['prob'] = df_risks.values
    df_risks = df_risks_
    df_risks['day'] = np.ones(len(df_risks))
    df_risks['timestamp'] = pd.to_datetime(df_risks[['year', 'month', 'day']])
    df_risks = df_risks.drop(columns=['year', 'month', 'day'])
    if cfg.eval.interpolation_res is not None:
        logging.info(f"Interpolating to {cfg.eval.interpolation_res} degrees resolution")
        df_risks = interpolate(df_risks, cfg)

    logging.info(f"Preparing format for dumping")
    lat_axis = np.sort(df_risks.lat.unique())
    lon_axis = np.sort(df_risks.lon.unique())
    dlat = np.unique(np.diff(lat_axis))[0]
    dlon = np.unique(np.diff(lon_axis))[0]
    
    lower_left = list(zip(df_risks['lon'], df_risks['lat']))
    upper_right = list(zip(df_risks['lon'] + dlon, df_risks['lat'] + dlat))
    lower_right = list(zip(df_risks['lon'] + dlon, df_risks['lat']))
    upper_left = list(zip(df_risks['lon'], df_risks['lat'] + dlat))
    df_risks['geom'] = list(zip(lower_left, upper_left, upper_right, lower_right))
    logging.info(f"Casting to geopandas")
    df = df_risks
    geometry = [Polygon(xy) for xy in df.geom]
    df = df.drop(['geom'], axis=1)
    df = df.drop(columns=['lat', 'lon'])
    gdf = GeoDataFrame(df, crs="EPSG:4326", geometry=geometry)
    if cfg.eval.save_kml:
        fiona.supported_drivers['KML'] = 'rw'
        gdf.to_file(cfg.eval.output_file + '.kml', driver='KML')
        logging.info(f"Saved to {cfg.eval.output_file}.kml")
        logging.info(f"Total time spent {time.process_time() - start_time} seconds")
    else:
        gdf.to_csv(cfg.eval.output_file + '.csv')
        logging.info(f"Saved to {cfg.eval.output_file}.csv")
        logging.info(f"Total time spent {time.process_time() - start_time} seconds")



@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs/infer_configs"), config_name="risk_estimation_20_test")
def main(cfg: DictConfig):    
    risk_estimation(cfg)
    logging.info('Risks are estimated finished!')


if __name__ == "__main__":      
    main()