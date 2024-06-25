import xarray as xr
import os
from tqdm import tqdm
import logging
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path=os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "configs"), config_name="bias_correction.yaml")
def run_zarr_load(cfg: DictConfig):
    
    TASK = cfg.load.task
    # Define time period for analysis
    YEAR_START = cfg.load["year_start"]
    YEAR_END = cfg.load["year_end"]

    # Years range
    years = range(YEAR_START, YEAR_END + 1)

    # Path to save files
    PATH = os.path.join(cfg.load.save_path, TASK, cfg.load.local_path)
    os.makedirs(PATH, exist_ok=True)

    # # vars = cfg.load["single_level_vars"]
    
    logging.info("ERA5 download script started.")
    # Access ERA5 from WeatherBench2
    data = xr.open_zarr(
        "gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2"
    )
    logging.info("Zarr opened.")
    
    if TASK == "era5-eobs":
        vars = [
                # "2m_temperature",
                # "minimum_temperature",
                # "maximum_temperature",
                # "rainfall",
                "land_sea_mask",
                "geopotential_at_surface"
        ]
        
        # Transform longitude to fit with other data
        data.coords["longitude"] = (data.coords["longitude"] + 180) % 360 - 180
        data = data.sortby(data.longitude)

        # Get eobs bounds
        top = cfg.load.eobs_bounds.top
        bottom = cfg.load.eobs_bounds.bottom
        left = cfg.load.eobs_bounds.left
        right = cfg.load.eobs_bounds.right
        
        # Clip source data with e-obs boundaries
        data = data.sel(latitude=slice(top,bottom), longitude=slice(left,right))
    
    elif TASK == "cmip6-era5":
            vars = [
                    # "2m_temperature",
                    # "total_precipitation",
                    # "10m_u_component_of_wind",
                    # "10m_v_component_of_wind",
                    # "surface_pressure",
                    "land_sea_mask",
                    "geopotential_at_surface"
                ]

    logging.info("Starting download...")
    const_var = []
    
    # Loop over variables
    for var in vars:
        if var in cfg.load["constants"]:
            logging.info(f"Variable: {var}")
            const_var.append(data[var])
        
        else:
            for year in tqdm(years):
                logging.info(f"Variable: {var}; Year: {year}")
                
                if var=="2m_temperature":
                    data_var = data[var].sel(time=str(year))
                    if TASK=="cmip6-era5":
                        data_sel = data_var.resample(time='D').mean(dim=["time"])
                        # data_sel = data_var.resample(time='3H').first()
                    elif TASK=="era5-eobs":
                        data_sel = data_var.resample(time='D').mean(dim=["time"])
                    # # Adjust the path and save to zarr
                    zarr_path = os.path.join(PATH, var, "{}_{}.zarr".format(var, year))
                    data_sel.to_zarr(zarr_path, mode='w')
                    
                elif var== "minimum_temperature":   #"era5-eobs"
                    data_var = data["2m_temperature"].sel(time=str(year))
                    data_sel = data_var.resample(time='D').min(dim=['time'])
                    # Adjust the path and save to zarr
                    zarr_path = os.path.join(PATH, var, "{}_{}.zarr".format(var, year))
                    data_sel.to_zarr(zarr_path, mode='w')
                    
                elif var== "maximum_temperature":     #"era5-eobs"
                    data_var = data["2m_temperature"].sel(time=str(year))
                    data_sel = data_var.resample(time='D').max(dim=['time'])
                    # Adjust the path and save to zarr
                    zarr_path = os.path.join(PATH, var, "{}_{}.zarr".format(var, year))
                    data_sel.to_zarr(zarr_path, mode='w')
                    
                elif var=="total_precipitation":    #"cmip6-era5"
                    data_var = data[var].sel(time=str(year))
                    data_sel = data_var.resample(time='D').mean(dim='time')
                    # Sum yields timestamps [0, 3, .., 21], corresponds to [1.30, 4.30...] at CMIP
                    # data_sel = data_var.resample(time='3H').sum(dim='time')
                    # Adjust the path and save to zarr
                    zarr_path = os.path.join(PATH, var, "{}_{}.zarr".format(var, year))
                    data_sel.to_zarr(zarr_path, mode='w')
                    
                elif var in ["10m_u_component_of_wind", "10m_v_component_of_wind", "surface_pressure"]: #"cmip6-era5"
                    data_var = data[var].sel(time=str(year))
                    data_sel = data_var.resample(time='D').mean(dim=["time"])
                    # data_sel = data_var.resample(time='3H').first()
                    # # Adjust the path and save to zarr
                    zarr_path = os.path.join(PATH, var, "{}_{}.zarr".format(var, year))
                    data_sel.to_zarr(zarr_path, mode='w')

                elif var == "rainfall":   #"era5-eobs"
                    data_var = data["total_precipitation"].sel(time=str(year))
                    data_sel = data_var.resample(time='D').sum(dim=["time"])  ## TODO check sum
                    # # Adjust the path and save to zarr
                    zarr_path = os.path.join(PATH, var, "{}_{}.zarr".format(var, year))
                    data_sel.to_zarr(zarr_path, mode='w')
                    
                elif var == "wind_speed_mean":   #"era5-eobs" 
                    # TODO
                    data_var = data["total_precipitation"].sel(time=str(year))
                    data_sel = data_var.resample(time='D').sum()
                    # # Adjust the path and save to zarr
                    zarr_path = os.path.join(PATH, var, "{}_{}.zarr".format(var, year))
                    data_sel.to_zarr(zarr_path, mode='w')
                
                else:
                    print(f"For variable {var} procedure not implemented")
    
    # Save constants.
    # Be aware, that data from "constants.nc" goes to processed .npy files. To get rid of this, change the name somehow
    const_path = os.path.join(PATH, "constants.nc")
    ds_const = xr.combine_by_coords(const_var)
    print(ds_const)
    # Add latitude as new variable
    ds_const["lat_grid"] = xr.DataArray(data = np.array([ds_const['latitude']]*len(ds_const["longitude"])).T, coords=ds_const.coords)
    ds_const.to_netcdf(const_path, mode='w')     
          
    logging.info("Download finished.")

if __name__ == "__main__":
    run_zarr_load()
