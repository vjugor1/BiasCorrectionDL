import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append(os.getcwd())
import hydra
import logging
import glob
from datetime import datetime
import pandas as pd
import warnings
from src.utils.norm_values import *
warnings.filterwarnings('ignore')

class PlotGenerator:

    def __init__(self, cfg):
        self.cfg = cfg
        if not os.path.exists(cfg.visual.plots_dir):
            os.makedirs(cfg.visual.plots_dir)

    def load_gt_data(self, var, date, save_gt_plot=False):

        file_paths =  glob.glob(self.cfg.visual.gt_data_folder + '/*' f'{var}_ens_mean_0.25deg*')
        assert os.path.isfile(file_paths[0])
        self.gt_data_arr = xr.open_mfdataset(file_paths)
        index = date_to_index(date, self.gt_data_arr['time'].to_numpy())
        half_w = self.cfg.train.time_agg_window//2
        i = 1 if self.cfg.train.time_agg_window % 2 == 0 else 0
        q = self.cfg.visual.quantile_gt

        self.gt_data_arr = self.gt_data_arr.sel(time=slice(self.gt_data_arr['time'][index-half_w],
                                                           self.gt_data_arr['time'][index+half_w - i]))
        self.gt_data_arr = self.gt_data_arr.sel(latitude=slice(self.cfg.visual.vis_lat_min,
                                                               self.cfg.visual.vis_lat_max))
        self.gt_data_arr = self.gt_data_arr.sel(longitude=slice(self.cfg.visual.vis_lon_min,
                                                               self.cfg.visual.vis_lon_max))

        if save_gt_plot:
            gt_arr = np.quantile(self.gt_data_arr[var].to_numpy(), q=q, method='weibull', axis=0)
            self.plot_map(gt_arr, self.gt_data_arr['latitude'], self.gt_data_arr['longitude'], var)


    def load_cmip_data(self):
        self.time_coords = np.load(os.path.join(self.cfg.train.data_dir, 'time.npy')).astype('datetime64[D]')
        self.lat_coords = np.load(os.path.join(self.cfg.train.data_dir, 'lat.npy'))
        self.lon_coords = np.load(os.path.join(self.cfg.train.data_dir, 'lon.npy'))
        var_data = np.empty(
            (len(self.cfg.process.variables), len(self.time_coords), len(self.lat_coords), len(self.lon_coords)),
            dtype=np.float16)
        
        var = self.cfg.visual.vis_cmip_var
        var_data = np.load(os.path.join(self.cfg.train.data_dir, var + f'_{self.cfg.process.precision}.npy'))

        self.var_data = var_data.astype(np.float32)
        self.crop_cmip_data()

    def crop_cmip_data(self):
        half_side = self.cfg.half_side_size
        self.lat_min_idx = np.searchsorted(self.lat_coords, self.cfg.visual.vis_lat_min)
        self.lat_max_idx = np.searchsorted(self.lat_coords, self.cfg.visual.vis_lat_max)
        self.lon_min_idx = np.searchsorted(self.lon_coords, self.cfg.visual.vis_lon_min)
        self.lon_max_idx = np.searchsorted(self.lon_coords, self.cfg.visual.vis_lon_max)
        
        self.lat_coords = self.lat_coords[self.lat_min_idx: self.lat_max_idx]
        self.lon_coords = self.lon_coords[self.lon_min_idx: self.lon_max_idx]
        logging.info(f"Lat : {min(self.lat_coords)} - {max(self.lat_coords)}, len {len(self.lat_coords)}")
        logging.info(f"Lon: {min(self.lon_coords)} - {max(self.lon_coords)}, len {len(self.lon_coords)}")
        self.var_data = self.var_data[:,
                                        self.lat_min_idx: self.lat_max_idx,
                                        self.lon_min_idx: self.lon_max_idx
                                        ]

    def get_cmip_by_date(self, date):
        index = date_to_index(date, self.time_coords)
        index_window = slice(index - self.cfg.train.time_agg_window//2, index + self.cfg.train.time_agg_window//2)
        self.var_data = self.var_data[index_window, :, :]
        self.time_coords = self.time_coords[index_window]
        assert len(self.time_coords) == self.var_data.shape[0]
        logging.info(f"Shape with time limits {self.var_data.shape}")


    def resample_to_gt(self):
        data_xr = xr.DataArray(self.var_data, 
        coords={'time': self.time_coords, 'latitude': self.lat_coords,'longitude': self.lon_coords,}, 
        dims=[ "time", "latitude", "longitude",])
        new_lons = np.arange(self.gt_data_arr["longitude"].min(), self.gt_data_arr["longitude"].max(), self.cfg.visual.gt_res)
        new_lats = np.arange(self.gt_data_arr["latitude"].min(), self.gt_data_arr["latitude"].max(), self.cfg.visual.gt_res)
        self.data_xr_interpolated = data_xr.interp(longitude=new_lons, latitude=new_lats)


    def plot_map(self, arr, lat, lon, name):
        fig, ax = plt.subplots(figsize=(12, 12))
        if self.cfg.visual.vis_eobs_var == 'fg':
            img = ax.imshow(arr, interpolation='nearest', extent=[lon.min(), lon.max(),
                                                                lat.max(), lat.min()], cmap='bwr', vmin=-10, vmax=6)
        if self.cfg.visual.vis_eobs_var == 'tn' or self.cfg.visual.vis_eobs_var == 'tx':
            img = ax.imshow(arr, interpolation='nearest', extent=[lon.min(), lon.max(),
                                                                lat.max(), lat.min()], cmap='bwr', vmin=-15, vmax=10)
        plt.gca().invert_yaxis()
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        fig.colorbar(img, cax=cax)
        savepath = os.path.join(self.cfg.visual.plots_dir, f'{name}.png')
        plt.savefig(savepath, dpi=200) 

    def load_eval_data(self):
        self.eval_df = pd.read_csv(self.cfg.visual.path_to_predictions) 
        self.eval_df = self.eval_df[self.eval_df['lat'] > self.cfg.visual.vis_lat_min]
        self.eval_df = self.eval_df[self.eval_df['lat'] < self.cfg.visual.vis_lat_max]
        self.eval_df = self.eval_df[self.eval_df['lon'] > self.cfg.visual.vis_lon_min]
        self.eval_df = self.eval_df[self.eval_df['lon'] <  self.cfg.visual.vis_lon_max]

        reshaped = self.eval_df.prediction.values.reshape(len(self.eval_df.date.unique()), len(self.eval_df.lat.unique()), len(self.eval_df.lon.unique()))
        data_xr = xr.DataArray(reshaped, 
                               coords={'time': np.sort(self.eval_df.date.unique()),
                                       'latitude': np.sort(self.eval_df.lat.unique()),
                                       'longitude': np.sort(self.eval_df.lon.unique())}, 
                                dims=["time", "latitude", "longitude"])
        
        new_lons = np.arange(self.gt_data_arr["longitude"].min(), self.gt_data_arr["longitude"].max(), self.cfg.visual.gt_res)
        new_lats = np.arange(self.gt_data_arr["latitude"].min(), self.gt_data_arr["latitude"].max(), self.cfg.visual.gt_res)
        self.eval_data_xr_interpolated = data_xr.interp(longitude=new_lons, latitude=new_lats)[0]


    def plot_eval_gt_diff(self):
        self.load_gt_data(self.cfg.visual.vis_eobs_var, self.cfg.visual.timestamp)
        self.load_eval_data()
        eval_arr = self.eval_data_xr_interpolated.data
        gt_arr = self.gt_data_arr.to_array().data[0, :, :eval_arr.shape[0],  :eval_arr.shape[1]]
        gt_arr = np.quantile(gt_arr, q=self.cfg.visual.quantile_gt, method='weibull', axis=0)
        diff = gt_arr - eval_arr
        logging.info(f"eval sum {np.nansum(np.abs(diff))}")
        self.plot_map(
                      diff,
                      self.gt_data_arr["latitude"][:eval_arr.shape[0]],
                      self.gt_data_arr["longitude"][:eval_arr.shape[1]],
                      'diff_eval')
        

    def plot_cmip_gt_diff(self):
        self.load_gt_data(self.cfg.visual.vis_eobs_var, self.cfg.visual.timestamp)
        self.load_cmip_data()
        self.get_cmip_by_date(self.cfg.visual.timestamp)
        self.resample_to_gt()

        variables_order_cmip = self.cfg.process.variables
        if self.cfg.cmip_type == "CMIP6":
            mean = mean_channels_cmip6[variables_order_cmip.index(self.cfg.visual.vis_cmip_var)]
            std = std_channels_cmip6[variables_order_cmip.index(self.cfg.visual.vis_cmip_var)]
        elif self.cfg.cmip_type == "CMIP5":
            mean = mean_channels_cmip5[variables_order_cmip.index(self.cfg.visual.vis_cmip_var)]
            std = std_channels_cmip5[variables_order_cmip.index(self.cfg.visual.vis_cmip_var)]
        else:
            raise NotImplementedError(f"No norm values for {self.cfg.cmip_type}")
        # cmip_arr = (self.data_xr_interpolated.data * 20.841803) +  280.37646 - 271.15
        cmip_arr = (self.data_xr_interpolated.data * std) + mean
        if (self.cfg.visual.vis_cmip_var == "tasmax") or (self.cfg.visual.vis_cmip_var == "tasmin"):
            cmip_arr = cmip_arr# - 273.15
        gt_arr = self.gt_data_arr.to_array().data[0, :, :cmip_arr.shape[1],  :cmip_arr.shape[2]]
        cmip_arr = np.quantile(cmip_arr, q=self.cfg.visual.quantile_cmip, method='weibull', axis=0)
        gt_arr = np.quantile(gt_arr, q=self.cfg.visual.quantile_gt, method='weibull', axis=0)
        diff = gt_arr - cmip_arr
        logging.info(f"cmip sum {np.nansum(np.abs(diff))}")

        self.plot_map(
                      diff,
                      self.gt_data_arr["latitude"][:cmip_arr.shape[0]],
                      self.gt_data_arr["longitude"][:cmip_arr.shape[1]],
                      'diff_cmip')

def date_to_index(date, dates_array):
    date = f'{date} 00:00:00'
    date = np.datetime64(date)
    date_index = np.searchsorted(dates_array , date)
    return date_index



@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="cmip6_GhostWindNet27.yaml")
def main(cfg):    
    PT = PlotGenerator(cfg)
    PT.plot_cmip_gt_diff()
    PT.plot_eval_gt_diff()


if __name__ == "__main__":      
    sys.argv.append('hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()