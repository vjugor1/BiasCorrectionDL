import numpy as np 
import pandas as pd
import polars
import unittest

import os, shutil
import sys
sys.path.append(os.path.join(os.getcwd()))

import hydra
from omegaconf import DictConfig

from src.regression import data_load as dl
from src.data_assemble.assemble_target import stations_to_data_grid
from test.dummy_data import dummy_climate

def test_pre_prepare_target():
    pass
@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs/unit_test_configs"), config_name="dummy_data_leap.yaml")
class TestTarget(unittest.TestCase):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
    def test_target(self):
        print("Target pixels")
        cfg = self.cfg
        #create dummy data
        data, data_xr, time_coords, lat_coords, lon_coords, climate_vars, df_data, stations_df, cfg = dummy_climate(number_of_days=4)

        indexes_st = {row[0]: (np.argmin(np.abs(lat_coords - row[1])), np.argmin(np.abs(lat_coords - row[2]))) for row in stations_df.iter_rows()}
        
        to_grid = stations_to_data_grid(data_xr, stations_df)
        stations_check = []
        for row in to_grid.iter_rows():
            stations_check.append((row[1] == indexes_st[row[0]][0]) and (row[2] == indexes_st[row[0]][1]))

        self.assertTrue(all(stations_check), "Found pixels does not pass the test")

        df = df_data.join(to_grid, on='station_name', how='left')
        df = df.select(polars.col(["time", "y", "lat", "lon"])).drop_nulls()
        pass

if __name__ == '__main__':
    unittest.main()