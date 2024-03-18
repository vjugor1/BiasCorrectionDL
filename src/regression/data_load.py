import sys,os
sys.path.append(os.getcwd())
from src.utils.data_utils import round_to_closest_indices, make_padding
import logging
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
import yaml
import hashlib
from numpy.lib.stride_tricks import sliding_window_view
from datetime import datetime
import time
import polars
import gc
from functools import partial
import glob 
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def clean_start():
    for f in glob.glob("tmp_Q_t*"):
        os.remove(f)



class DataPreLoader:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg    
        logging.info("--- MULTI REGRESSION ---")
        assert (cfg.process.precision == 16 and not cfg.train.normalize) or (cfg.process.precision == 32 and cfg.train.normalize), \
        ''' 16 bit is already normalized. 32 bit is not normalized'''        
        self.generate_hash()       
        self.dataset_torch = self.load_climate_data()
        self.dataset_torch = self.time_crop(self.dataset_torch)

        clean_start()
        if self.cfg.train.make_tmp_target_file:
            if self.data_exists():
                self.load_data()
            else:
                self.prepare_target_df()
                self.target_df_to_array()
                self.save_data()
        else:   
            self.prepare_target_df()
            self.target_df_to_array()
        self.log_data()


    def load_climate_data(self):
        dtype = np.float16 if self.cfg.process.precision == 16 else np.float32
        self.time_coords = np.load(os.path.join(self.cfg.train.data_dir, 'time.npy')).astype('datetime64[D]')
        self.lat_coords = np.load(os.path.join(self.cfg.train.data_dir, 'lat.npy'))
        self.lon_coords = np.load(os.path.join(self.cfg.train.data_dir, 'lon.npy'))

        var_data = np.empty(
            (len(self.cfg.train.variables), len(self.time_coords), len(self.lat_coords), len(self.lon_coords)),
            dtype=dtype)
        for i, var in enumerate(self.cfg.train.variables):
            var_data[i] = np.load(os.path.join(self.cfg.train.data_dir, var + f'_{self.cfg.process.precision}.npy'))
        logging.info(f"CMIP data loaded {var_data.shape}")

        if self.cfg.train.spatial_crop:
            var_data = self.spatial_crop(var_data)
            logging.info(f"Cropped data shape {var_data.shape}")

        else:
            var_data, self.shift = make_padding(var_data, self.cfg.half_side_size)
            logging.info(f"Padded data shape {var_data.shape}")

        # var_data_torch = torch.from_numpy(var_data).half() if self.cfg.process.precision == 16 else torch.from_numpy(var_data)
        var_data_torch = torch.from_numpy(var_data).type(torch.float32)

        return var_data_torch

    def spatial_crop(self, var_data): 
        half_side = self.cfg.half_side_size
        self.lat_min_idx = np.searchsorted(self.lat_coords, self.cfg.train.lat_min)
        self.lat_max_idx = np.searchsorted(self.lat_coords, self.cfg.train.lat_max)
        self.lon_min_idx = np.searchsorted(self.lon_coords, self.cfg.train.lon_min)
        self.lon_max_idx = np.searchsorted(self.lon_coords, self.cfg.train.lon_max)
        
        self.lat_coords_crop = self.lat_coords[self.lat_min_idx: self.lat_max_idx]
        self.lon_coords_crop = self.lon_coords[self.lon_min_idx: self.lon_max_idx]
        logging.info(f"Lat : {min(self.lat_coords_crop)} - {max(self.lat_coords_crop)}, len {len(self.lat_coords_crop)}")
        logging.info(f"Lon: {min(self.lon_coords_crop)} - {max(self.lon_coords_crop)}, len {len(self.lon_coords_crop)}")
        logging.info(f"Lat indexes: {(self.lat_min_idx)} - {(self.lat_max_idx)}, len {len(self.lat_coords_crop)}")
        logging.info(f"Lon indexes: {(self.lon_min_idx)} - {(self.lon_max_idx)}, len {len(self.lon_coords_crop)}")
        if  (self.lat_min_idx < half_side) or \
            (self.lon_min_idx < half_side) or \
            (len(self.lon_coords) - self.lon_max_idx < half_side) or \
            (len(self.lat_coords) - self.lat_max_idx < half_side):
            logging.info(f"Pad + crop")

            var_data, self.shift = make_padding(var_data, self.cfg.half_side_size)
            var_data = var_data[
                                :,
                                :,
                                self.lat_min_idx: self.lat_max_idx + 2*half_side + 1,
                                self.lon_min_idx: self.lon_max_idx + 2*half_side + 1
                                ]
        else:
            logging.info(f"Just crop")
            var_data = var_data[
                                :,
                                :,
                                self.lat_min_idx - half_side: self.lat_max_idx + half_side + 1,
                                self.lon_min_idx - half_side: self.lon_max_idx + half_side + 1
                                ]
        return var_data

    def time_crop(self, var_data):
        start_date = datetime.strptime(self.cfg.train.start_time, '%Y-%m-%d').date()
        end_date = datetime.strptime(self.cfg.train.end_time, '%Y-%m-%d').date()
        start_index = self.time_coords.searchsorted(start_date)
        end_index = self.time_coords.searchsorted(end_date)
        var_data = var_data[:, start_index:end_index, :, :]
        self.time_coords = self.time_coords[start_index:end_index]
        assert len(self.time_coords) == var_data.shape[1]
        logging.info(f"Shape with time limits {var_data.shape}")
        return var_data
    

    #### Target prep
    def time_to_data_grid(self, target_df):
        start_time = time.process_time()   
        dates = target_df["time"].to_numpy()
        y = target_df["y"].to_numpy()
        values = round_to_closest_indices(dates, self.time_coords) # time into inds
        target_df = target_df.with_columns(
                            polars.Series(name="time", values=values),
                            polars.Series(name="y", values=y),
                            )
        logging.info(f"Time align took {time.process_time() - start_time} seconds")
        return target_df


    def stations_to_data_grid(self, stations_df: polars.DataFrame) -> polars.DataFrame:
        """ maps stations to the data grid pixels """
        start_time = time.process_time()   
        lat = stations_df["lat"].to_numpy() 
        lon = stations_df["lon"].to_numpy()
        lat_vector = round_to_closest_indices(lat, self.lat_coords)
        lon_vector = round_to_closest_indices(lon, self.lon_coords)
        stations_df = stations_df.with_columns(
                            [
                             polars.Series(name="lat", values=lat_vector),
                             polars.Series(name="lon", values=lon_vector)
                            ])
        logging.info(f"Closest pixel search took {time.process_time() - start_time} seconds")
        return stations_df
    

    def prepare_target_df(self):
        target_df = polars.read_parquet(os.path.join(self.cfg.train.data_dir, self.cfg.train.target_data_file))
        logging.info(f"Records before preparation {len(target_df)}")
        start_date = pd.to_datetime(self.cfg.train.start_time)
        end_date = pd.to_datetime(self.cfg.train.end_time)
        logging.info(f"Target time bounds before filter {target_df['time'].min()}, {target_df['time'].max()}")
        target_df = target_df.filter((polars.col('time') >= start_date) & (polars.col('time') < end_date))
        logging.info(f"Target time bounds after filter {target_df['time'].min()}, {target_df['time'].max()}")
        logging.info(f"Data time bounds {self.time_coords.min()}, {self.time_coords.max()}")
        logging.info(f"Stations before aggregation: {target_df.n_unique(subset=['lat', 'lon'])}")
        target_df = self.time_to_data_grid(target_df)
        target_df = self.stations_to_data_grid(target_df)

        target_df = (target_df
                    .lazy()        
                    .sort("time")
                    .groupby(["lat", "lon", "time"])
                    .agg(
                        [
                         polars.col('y').quantile(0.65).alias("y"),
                        ])
                    .collect())
        
        target_df = (target_df
                    .lazy()        
                    .sort("time")
                    .groupby(["lat", "lon"])
                    .agg(
                        [
                         polars.col("time"),
                         polars.col('y'),
                        ])
                    .collect())
        self.target_df = target_df.drop_nulls()
        logging.info(f"Stations after aggregation: {len(target_df)}")


    def target_df_to_array(self):
        start_time = time.process_time()   
        split_date = datetime.strptime(self.cfg.train.start_of_test, '%Y-%m-%d').date()
        split_index = self.time_coords.searchsorted(split_date)
        targets_list = []
        drop_dict = {}
        total = 0
        for target_df_row in self.target_df.rows():
            lat, lon, dates, y = target_df_row
            y = np.array(y)
            dates = np.array(dates)
            res = self.stations_filter(lat, lon, dates, y, self.cfg.target_type)
            if res != True:
                if res not in drop_dict:
                    drop_dict[res] = 1
                else:
                    drop_dict[res] += 1
                continue
            targets_list.append(self.pixel_aggregation(lat, lon, dates, y))
            total += 1
        logging.info(f"Pixel loop took {time.process_time() - start_time} seconds, droped {drop_dict}")
        logging.info(f"Stations finally: {total}")
        target_array = np.concatenate(targets_list, axis=1)
        del targets_list
        target_array = target_array[:, ::self.cfg.train.time_freq]

        if self.cfg.train.spatial_crop:
            target_array[0, :] += self.shift[0] - self.lat_min_idx #lat
            target_array[1, :] += self.shift[1] - self.lon_min_idx #lon
        else:
            target_array[0, :] += self.shift[0] #lat
            target_array[1, :] += self.shift[1] #lon
        self.train_data_idxs = target_array[:, target_array[2, :] < split_index]
        self.test_data_idxs = target_array[:, target_array[2, :] > split_index]
        self.test_data_idxs = self.test_data_idxs[:, self.test_data_idxs[2, :] < len(self.time_coords)]
        logging.info(f'Records prepared train {self.train_data_idxs.shape[1]}')
        logging.info(f'Records prepared test {self.test_data_idxs.shape[1]}')
        gc.collect()

    def stations_filter(self, lat, lon, dates, y, target_type): 
        if len(y) < max(self.cfg.train.time_agg_window, self.cfg.time_window):
            return "too short"
        if self.cfg.train.spatial_crop: 
            if lat < self.lat_min_idx or lat > self.lat_max_idx or lon < self.lon_min_idx or lon > self.lon_max_idx:
                # print(f"drop {lat, lon}")
                return "out of train area"

        if target_type == 'temp_c':
            if np.count_nonzero(y < -35)/y.size > 0.9:
                return "low temp"
            if np.count_nonzero(y > 40)/y.size > 0.5:
                return "high temp" 
        elif target_type == 'wind_ms':
            if np.count_nonzero(y < 2)/y.size > 0.9:
                return "low speed"
            if np.count_nonzero(y > 16)/y.size > 0.5:
                return "high speed"
        else: 
            raise NotImplementedError
        return True


    def pixel_aggregation(self, lat, lon, dates, y):
        # aggregate target with given time_agg_window 
        time_positions_m = np.array([d.astype(object).month for d in self.time_coords[dates]])
        time_positions_days =  np.array([d.astype(object).day for d in self.time_coords[dates]])
        time_positions = (time_positions_m * 30.5 + time_positions_days)/365
        time_positions_m = time_positions_m/12
        assert len(time_positions) == len(dates)
        y_agg_quantlies = np.quantile(sliding_window_view(y, window_shape=self.cfg.train.time_agg_window), 
                        q=[0.96, 0.85, 0.70, 0.50, 0.25, 0.15, 0.05],
                        axis = 1,
                        method='weibull')
        i = 1 if self.cfg.train.time_agg_window % 2 == 0 else 0
        if self.cfg.time_window > self.cfg.train.time_agg_window:
            # clip dates according to time_window
            dates = dates[self.cfg.time_window//2: len(dates)-self.cfg.time_window//2 + i] 
            time_positions = time_positions[self.cfg.time_window//2: len(time_positions)-self.cfg.time_window//2 + i] 
            time_positions_m = time_positions_m[self.cfg.time_window//2: len(time_positions_m)-self.cfg.time_window//2 + i] 

            y_agg_quantlies = y_agg_quantlies[self.cfg.time_window-self.cfg.train.time_agg_window:
                                              len(y_agg_quantlies) + self.cfg.train.time_agg_window - self.cfg.time_window - 1]
        else:
            dates = dates[self.cfg.train.time_agg_window//2: len(dates)-self.cfg.train.time_agg_window//2 + i] 
            time_positions = time_positions[self.cfg.train.time_agg_window//2: len(time_positions)-self.cfg.train.time_agg_window//2 + i] 
            time_positions_m = time_positions_m[self.cfg.train.time_agg_window//2: len(time_positions_m)-self.cfg.train.time_agg_window//2 + i] 
             # y_agg_quantlies not changed

        assert len(time_positions) == len(dates)
        mask = dates > self.cfg.time_window//2+1
        dates = dates[mask]
        time_positions = time_positions[mask]
        time_positions_m = time_positions_m[mask]
        y_agg_quantlies = y_agg_quantlies[:, mask]

        lat_position = self.lat_coords[lat]/90
        lon_position = self.lon_coords[lon]/180
        target_array = np.stack([np.full(len(dates), lat),
                                 np.full(len(dates), lon),
                                 dates,
                                 time_positions,
                                 time_positions_m,
                                 np.full(len(dates), lat_position),
                                 np.full(len(dates), lon_position),
                                 ])
        
        target_array = np.concatenate((target_array, y_agg_quantlies), axis=0)
        return target_array
    
    ### Utils for preload
    def generate_hash(self):
        config_str = yaml.dump(OmegaConf.to_yaml(self.cfg), sort_keys=True)
        self.config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()

    def data_exists(self):
        return os.path.isfile(f'tmp_Q_train_{self.config_hash}.npz')

    def save_data(self):
        np.savez_compressed(f'tmp_Q_train_{self.config_hash}.npz', self.train_data_idxs)
        np.savez_compressed(f'tmp_Q_test_{self.config_hash}.npz', self.test_data_idxs)

    def load_data(self):
        self.train_data_idxs = np.load(f'tmp_Q_train_{self.config_hash}.npz')['arr_0']
        self.test_data_idxs = np.load(f'tmp_Q_test_{self.config_hash}.npz')['arr_0']

    def log_data(self):
        logging.info(f"Train size: {self.train_data_idxs.shape[1]}, test size: {self.test_data_idxs.shape[1]}")
        logging.info(f"Target min: {self.train_data_idxs[7, :].min()}, target max: {self.train_data_idxs[7, :].max()}")
        logging.info(f"Target mean: {self.train_data_idxs[7, :].mean()}, target std: {self.train_data_idxs[7, :].std()}")
        logging.info(f"Balance train: {self.get_class_balance(self.train_data_idxs[7, :])}, balance test:{self.get_class_balance(self.test_data_idxs[7, :])}")
        for i, var in enumerate(self.cfg.train.variables):
            logging.info(f"{var} mean: {self.dataset_torch[i].mean()}, std: {self.dataset_torch[i].std()}")

    def get_class_balance(self, target_array):
        positive = np.sum(target_array >= self.cfg.train.target_threshold)
        all = target_array.shape[0]
        return positive/all
            
    