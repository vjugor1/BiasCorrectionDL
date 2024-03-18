import numpy as np 
import pandas as pd
import unittest
import xarray as xr

import os, shutil
import sys
sys.path.append(os.path.join(os.getcwd()))

import hydra
from omegaconf import DictConfig

from src.regression.data_load import DataPreLoaderAlt
try:
    from test.dummy_data import dummy_climate
except ModuleNotFoundError:
    from dummy_data import dummy_climate

# @hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs/unit_test_configs"), config_name="dummy_data_leap.yaml")
# class TestClimatePadding(unittest.TestCase):
#     def __init__(self, cfg: DictConfig):
#         super().__init__()
#         self.cfg = cfg
# def test_climatedata_to_patches(self):


def assert_quadrants_coords(xr_quadrants, lat_lims, lon_lims):
    """Quadrants:
    II  |  I
    ________
    III | IV
    """
    qs_lon_lims = [(q.lon.data.min(), q.lon.data.max()) for q in xr_quadrants]
    qs_lat_lims = [(q.lat.data.min(), q.lat.data.max()) for q in xr_quadrants]
    lat_lims_middle = (lat_lims[0] + lat_lims[1]) / 2
    lon_lims_middle = (lon_lims[0] + lon_lims[1]) / 2
    
    #First quadrant check
    assert((qs_lon_lims[0][1] <= lon_lims[1]) and (qs_lon_lims[0][0] >= lon_lims_middle)), "First quadrant wrong Longitude limits"
    assert((qs_lat_lims[0][1] <= lat_lims[1]) and (qs_lat_lims[0][0] >= lat_lims_middle)), "First quadrant wrong Latitude limits"

    #Second quadrant check
    assert((qs_lon_lims[1][1] <= lon_lims_middle) and (qs_lon_lims[1][0] >= lon_lims[0])), "Second quadrant wrong Longitude limits"
    assert((qs_lat_lims[1][1] <= lat_lims[1]) and (qs_lat_lims[1][0] >= lat_lims_middle)), "Second quadrant wrong Latitude limits"

    #Third quadrant check
    assert((qs_lon_lims[2][1] <= lon_lims_middle) and (qs_lon_lims[2][0] >= lon_lims[0])), "Third quadrant wrong Longitude limits"
    assert((qs_lat_lims[2][1] <= lat_lims_middle) and (qs_lat_lims[2][0] >= lat_lims[0])), "Third quadrant wrong Latitude limits"

    #Fourth quadrant check
    assert((qs_lon_lims[3][1] <= lon_lims[1]) and (qs_lon_lims[3][0] >= lon_lims_middle)), "Third quadrant wrong Longitude limits"
    assert((qs_lat_lims[3][1] <= lat_lims_middle) and (qs_lat_lims[3][0] >= lat_lims[0])), "Third quadrant wrong Latitude limits"

def test_padding_correctness(padded_map, data_xr, xr_q, half_side_size):
    middle = data_xr.sel(lon=xr_q[0].lon.data[-1])
    top = data_xr.sel(lon=xr_q[1].lon.data[-1], lat=slice(0, 90))
    top = top.reindex(lat=list(reversed(top.lat)))
    bot = data_xr.sel(lon=xr_q[2].lon.data[-1], lat=slice(-90, 0))
    bot = bot.reindex(lat=list(reversed(bot.lat)))
    combined = np.concatenate((bot.data[..., -half_side_size:], middle.data, top.data[..., :half_side_size]), axis=-1)
    assert(combined.shape[-1] == padded_map.shape[-2]), "Wrong padded map shape"
    assert(np.allclose(padded_map[..., half_side_size-1] - combined, 0)), "values of padded map are wrong at Longitude=360"
    pass


def test_pad_climate_data(map_test=False):
    print("Padding")
    # cfg = self.cfg
    #create dummy data
    lat_lims = (-90, 90)
    lon_lims = (0, 360)
    half_side_size = 5
    if map_test:
        print("reading cmip sample")
        data_xr = xr.open_mfdataset('/app/wind/data/cmip/MRI-highres_H/pr_day_MRI-AGCM3-2-H_highresSST-future_r1i1p1f1_gn_20150101-20241231.nc')
        var = [k for k in data_xr.data_vars.keys()][-1]
        data_xr = data_xr[var].compute()
        print("cmip sample has been read")
    else:
        data, data_xr, time_coords, lat_coords, lon_coords, climate_vars, df_data, stations_df, cfg = dummy_climate(start_date='2020-12-02', number_of_days=10, lat_lims=lat_lims, lon_lims=lon_lims, dlat=0.56, dlon=1.5)
    # halfs = {"lat": data.shape[-2] // 2, "lon": data.shape[-1] // 2}
    xr_q, shift = DataPreLoaderAlt.extract_quadrants(data_xr)
    xr_q_borders = DataPreLoaderAlt.extrect_quadrant_borders(xr_q, half_side_size)
    assert_quadrants_coords(xr_q, lat_lims, lon_lims)
    
    padded_map = DataPreLoaderAlt.assemble_padded_map(xr_q, xr_q_borders, half_side_size)
    test_padding_correctness(padded_map, data_xr, xr_q, half_side_size)
    
    

if __name__ == "__main__":
    # map_test = True
    # test_pad_climate_data(map_test)
    map_test = False
    test_pad_climate_data(map_test)
    # unittest.main()