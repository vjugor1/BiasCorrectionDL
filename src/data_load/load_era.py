import xarray as xr
import os
from tqdm import tqdm
import logging
import sys

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path=os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "configs"), config_name="bias_correction.yaml")
def run_zarr_load(cfg: DictConfig):#vars, years, PATH):
    # Define time period for analysis
    YEAR_START = cfg.load["year_start"]
    YEAR_END = cfg.load["year_end"]

    # Years range
    years = range(YEAR_START, YEAR_END + 1)

    # Path to save files
    PATH = cfg.load["era_save_path"]
    os.makedirs(PATH, exist_ok=True)

    vars = cfg.load["single_level_vars"]
    
    logging.info("ERA5 download script started.")
    # Access ERA5 from WeatherBench2
    data = xr.open_zarr(
        "gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2"
    )
    logging.info("Zarr opened.")
    
    # Transform longitude to fit with other data
    # data.coords["longitude"] = (data.coords["longitude"] + 180) % 360 - 180
    # data = data.sortby(data.longitude)

    # Loop over variables
    logging.info("Starting download...")
    for year in tqdm(years):
        for var in vars:
            logging.info(f"Variable: {var}; Year: {year}")
            data_var = data[var].sel(time=str(year))
            
            if var=="2m_temperature":
                data_sel = data_var.resample(time='3H').first()
                # # Adjust the path and save to zarr
                zarr_path = os.path.join(PATH, var, "{}_{}.zarr".format(var, year))
                data_sel.to_zarr(zarr_path, mode='w')
                
                # data_sel = data_var.resample(time='3H').min(dim='time')s
                # # Adjust the path and save to zarr
                # zarr_path = os.path.join(PATH, "{}_min_{}.zarr".format(var, year))
                # data_sel.to_zarr(zarr_path, mode='w')
                
                # data_sel = data_var.resample(time='3H').max(dim='time')
                # # Adjust the path and save to zarr
                # zarr_path = os.path.join(PATH, "{}_max_{}.zarr".format(var, year))
                # data_sel.to_zarr(zarr_path, mode='w')
                
            elif var=="total_precipitation":
                data_sel = data_var.resample(time='3H').sum(dim='time')
                # Adjust the path and save to zarr
                zarr_path = os.path.join(PATH, var, "{}_{}.zarr".format(var, year))
                data_sel.to_zarr(zarr_path, mode='w')
                
            elif var in ["10m_u_component_of_wind", "10m_v_component_of_wind", "surface_pressure"]:
                data_sel = data_var.resample(time='3H').first()
                # # Adjust the path and save to zarr
                zarr_path = os.path.join(PATH, var, "{}_{}.zarr".format(var, year))
                data_sel.to_zarr(zarr_path, mode='w')
                
            else:
                print(f"For variable {var} procedure not implemented")
            
    logging.info("Download finished.")

if __name__ == "__main__":
    run_zarr_load()
