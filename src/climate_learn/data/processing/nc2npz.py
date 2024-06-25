# Standard library
import glob
import os

# Third party
import numpy as np
import xarray as xr
import netCDF4 as nc
from tqdm import tqdm
import xesmf
from omegaconf import DictConfig
from typing import Callable, Optional, Tuple, Union
import yaml


# Get eobs bounds
with open("/app/configs/load/era.yaml", 'r') as file:
    cfg = yaml.safe_load(file)
TOP = cfg["eobs_bounds"]["top"]
BOTTOM = cfg["eobs_bounds"]["bottom"]
LEFT = cfg["eobs_bounds"]["left"]
RIGHT = cfg["eobs_bounds"]["right"]
    
    
# Local application
from .era5_constants import (
    DEFAULT_PRESSURE_LEVELS as default_pressure_levels_era,
    NAME_TO_VAR as name_to_var_era,
    VAR_TO_NAME as var_to_name_era,
    CONSTANTS as constants_era
)

from .cmip6_constants import (
    DEFAULT_PRESSURE_LEVELS as default_pressure_levels_cmip,
    NAME_TO_VAR as name_to_var_cmip,
    VAR_TO_NAME  as var_to_name_cmip,
    CONSTANTS as constants_cmip
)

from .eobs_constants import (
    NAME_TO_VAR as name_to_var_eobs,
    VAR_TO_NAME  as var_to_name_eobs,
    CONSTANTS as constants_eobs
)

def nc2np(path,
          src,
          variables,
          years,
          save_dir,
          partition,
          num_shards_per_year,
          frequency,
          regridder,
          periodic,
          scale_factor,
          align_target
        ):
    if src=="era5":
        DEFAULT_PRESSURE_LEVELS=default_pressure_levels_era
        NAME_TO_VAR=name_to_var_era
        VAR_TO_NAME=var_to_name_era
        CONSTANTS=constants_era
    elif src=="cmip6":
        DEFAULT_PRESSURE_LEVELS=default_pressure_levels_cmip
        NAME_TO_VAR=name_to_var_cmip
        VAR_TO_NAME=var_to_name_cmip
        CONSTANTS=constants_cmip
    elif src=="eobs":
        NAME_TO_VAR=name_to_var_eobs
        VAR_TO_NAME=var_to_name_eobs
        CONSTANTS=constants_eobs
    else:
        print("Set one of sources from [era5, cmip6, eobs]")
        
    assert frequency in ["H", "3H", "D"]
    if frequency=="3H":
        OBJ_PER_YEAR = 365*8 #(3hr period yields 8 obj per day)
    elif frequency=="H":
        OBJ_PER_YEAR = 365*24
    elif frequency=="D":
        OBJ_PER_YEAR = 365

    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)

    if partition == "train":
        normalize_mean = {}
        normalize_std = {}
    climatology = {}

    constants_path = os.path.join(path, "constants.nc")
    constants_are_downloaded = os.path.exists(constants_path)

    if constants_are_downloaded:
        constants = xr.open_dataset(
            constants_path
        )
        # Avoid odd length of lat/lon
        if any(len(ax)%2==1 for ax in list(constants.coords)):
            regridder_const, _ = regrid(constants, periodic=periodic)
            constants = regridder_const(constants, keep_attrs=True)
        
        constant_fields = [VAR_TO_NAME[v] for v in CONSTANTS if v in VAR_TO_NAME.keys()]
        constant_values = {}
        for f in constant_fields:
            constant_values[f] = np.expand_dims(
                constants[NAME_TO_VAR[f]].to_numpy(), axis=(0, 1)
            ).repeat(OBJ_PER_YEAR, axis=0)
            if partition == "train":
                normalize_mean[f] = constant_values[f].mean(axis=(0, 2, 3))
                normalize_std[f] = constant_values[f].std(axis=(0, 2, 3))

    for year in tqdm(years):
        np_vars = {}
        # constant variables
        if constants_are_downloaded:
            for f in constant_fields:
                np_vars[f] = constant_values[f]

        # non-constant fields
        for var in variables:
            if src=="eobs":
                var = NAME_TO_VAR[var]
                ds = open_eobs(path, var)
            elif src=="cmip6":
                ds = open_cmip(path, var)
            elif src=="era5":
                ds = open_era(path, var)
                ds.transpose("time", "latitude", "longitude", ...)
            
            # Get variable name from data
            codes=list(ds.keys())
            code = [x for x in codes if (x not in ["lat_bnds", "lon_bnds", "time_bnds"])][0] 
            
            if frequency=="3H":
                if code in ["pr", "total_precipitation"]:
                    ds = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31")) 
                else:
                    ds = ds.sel(time=slice(f"{year}-01-01 01:00:00", f"{year+1}-01-01 00:00:00"))   
            else:
                ds = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
            try:
                ds = regridder(ds, keep_attrs=True)
            except ValueError:
                # Falls here due to variations in eobs data only
                regridder_current, _ = regrid(ds,
                                              periodic=periodic,
                                              scale_factor=scale_factor,
                                              align_target =align_target)
                ds = regridder_current(ds, keep_attrs=True)
            
            if len(ds[code].shape) == 3:  # surface level variables
                ds[code] = ds[code].expand_dims("val", axis=1)
                if code == "total_precipitation":  
                    ds[code] = ds[code] * 1000   #ERA precip: m --> mm

                elif code=="pr": #CMIP precipitation: kg/m2/s --> mm
                    if frequency=="3H":
                        ds[code] = ds[code]*60*60*3
                    elif frequency=="H":
                        ds[code] = ds[code]*60*60
                    elif frequency=="D":
                        ds[code] = ds[code]*60*60*24
                        
                elif code in ["tg", "tx", "tn"]: #EOBS temp: C --> K
                    ds[code] = ds[code] + 273.15
                
                # remove the last 24 hours if this year has 366 days
                np_vars[var] = ds[code].to_numpy()[:OBJ_PER_YEAR]

                if partition == "train":
                    # compute mean and std of each var in each year
                    # var_mean_yearly = np_vars[var].mean(axis=(0, 2, 3))
                    # var_std_yearly = np_vars[var].std(axis=(0, 2, 3))
                    var_mean_yearly = np.nanmean(np_vars[var], axis=(0, 2, 3))
                    var_std_yearly = np.nanstd(np_vars[var], axis=(0, 2, 3))
                    if var not in normalize_mean:
                        normalize_mean[var] = [var_mean_yearly]
                        normalize_std[var] = [var_std_yearly]
                    else:
                        normalize_mean[var].append(var_mean_yearly)
                        normalize_std[var].append(var_std_yearly)

                clim_yearly = np_vars[var].mean(axis=0)
                # clim_yearly = np.nanmean(np_vars[var], axis=0)
                if var not in climatology:
                    climatology[var] = [clim_yearly]
                else:
                    climatology[var].append(clim_yearly)

            else:  # pressure-level variables
                assert len(ds[code].shape) == 4
                all_levels = ds["level"][:].to_numpy()
                all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS)
                for level in all_levels:
                    ds_level = ds.sel(level=[level])
                    level = int(level)
                    # remove the last 24 hours if this year has 366 days
                    np_vars[f"{var}_{level}"] = ds_level[code].to_numpy()[
                        :OBJ_PER_YEAR
                    ]
                    
                    if partition == "train":
                        # compute mean and std of each var in each year
                        var_mean_yearly = np_vars[f"{var}_{level}"].mean(axis=(0, 2, 3))
                        var_std_yearly = np_vars[f"{var}_{level}"].std(axis=(0, 2, 3))
                        if f"{var}_{level}" not in normalize_mean:
                            normalize_mean[f"{var}_{level}"] = [var_mean_yearly]
                            normalize_std[f"{var}_{level}"] = [var_std_yearly]
                        else:
                            normalize_mean[f"{var}_{level}"].append(var_mean_yearly)
                            normalize_std[f"{var}_{level}"].append(var_std_yearly)

                    clim_yearly = np_vars[f"{var}_{level}"].mean(axis=0)
                    if f"{var}_{level}" not in climatology:
                        climatology[f"{var}_{level}"] = [clim_yearly]
                    else:
                        climatology[f"{var}_{level}"].append(clim_yearly)

        assert OBJ_PER_YEAR % num_shards_per_year == 0
        num_per_shard = OBJ_PER_YEAR // num_shards_per_year
        for shard_id in range(num_shards_per_year):
            start_id = shard_id * num_per_shard
            end_id = start_id + num_per_shard
            sharded_data = {k: np_vars[k][start_id:end_id] for k in np_vars.keys()}
            np.savez(
                os.path.join(save_dir, partition, f"{year}_{shard_id}.npz"),
                **sharded_data,
            )

    if partition == "train":
        for var in normalize_mean.keys():
            if not constants_are_downloaded or var not in constant_fields:
                normalize_mean[var] = np.stack(normalize_mean[var], axis=0)
                normalize_std[var] = np.stack(normalize_std[var], axis=0)

        for var in normalize_mean.keys():  # aggregate over the years
            if not constants_are_downloaded or var not in constant_fields:
                mean, std = normalize_mean[var], normalize_std[var]
                # var(X) = E[var(X|Y)] + var(E[X|Y])
                variance = (
                    (std**2).mean(axis=0)
                    + (mean**2).mean(axis=0)
                    - mean.mean(axis=0) ** 2
                )
                std = np.sqrt(variance)
                # E[X] = E[E[X|Y]]
                mean = mean.mean(axis=0)
                normalize_mean[var] = mean
                # if var == "total_precipitation":
                #     normalize_mean[var] = np.zeros_like(normalize_mean[var])
                normalize_std[var] = std

        np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
        np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)

    for var in climatology.keys():
        climatology[var] = np.stack(climatology[var], axis=0)
    climatology = {k: np.mean(v, axis=0) for k, v in climatology.items()}
    np.savez(
        os.path.join(save_dir, partition, "climatology.npz"),
        **climatology,
    )


def regrid(ds_in: xr.Dataset,
           periodic: bool,
           scale_factor: Optional[Union[int, float]]=1,
           align_target: Optional[str]=None
           ):
    if not align_target:
        ds_target=ds_in
    elif "cmip6-era5" in align_target:
        var=glob.glob(os.path.join(align_target, "*"))[0].split("/")[-1]
        ds_target = open_era(align_target, var)
    elif "era5-eobs" in align_target:
        var = glob.glob(os.path.join(align_target, "*"))[0].split("/")[-1]
        ds_target = open_era(align_target, var)

    if len(ds_target["latitude"])%2!=0:
        n_cells_lat=(len(ds_target["latitude"])-1)/scale_factor
    else:
        n_cells_lat=len(ds_target["latitude"])/scale_factor
    if periodic==True:
        n_cells_lon=len(ds_target["longitude"])/scale_factor
    else:
        n_cells_lon=(len(ds_target["longitude"])-1)/scale_factor

    lon_new = np.linspace(
        np.min(ds_target["longitude"].values),
        np.max(ds_target["longitude"].values),
        int(n_cells_lon))
    lat_new = np.linspace(
        np.min(ds_target["latitude"].values),
        np.max(ds_target["latitude"].values),
        int(n_cells_lat))

    grid_out = {'lon': lon_new, 'lat': lat_new}
    
    regridder = xesmf.Regridder(ds_in,
                                grid_out,
                                "bilinear",
                                periodic=periodic
                                )
    return regridder, grid_out


def convert_nc2npz(
    root_dir: str,
    save_dir: str,
    src: str,
    variables: list[str],
    start_train_year: int,
    start_val_year: int,
    start_test_year: int,
    end_year: int,
    num_shards: int,
    frequency: str,
    align_target: Optional[str]=None,
    scale_factor: Optional[int]=1,
    periodic: Optional[bool]=True
):
    assert (
        start_val_year > start_train_year
        and start_test_year > start_val_year
        and end_year > start_test_year
    )
    train_years = range(start_train_year, start_val_year)
    val_years = range(start_val_year, start_test_year)
    test_years = range(start_test_year, end_year)
    os.makedirs(save_dir, exist_ok=True)
    
    if src=="eobs":
        var = name_to_var_eobs[variables[0]]
        ds = open_eobs(root_dir, var)
    elif src=="era5":
        ds = open_era(root_dir, variables[0])
    elif src=="cmip6":
        ds = open_cmip(root_dir, variables[0])
        
    # create regridder and save lat/lon data
    regridder, grid_out = regrid(ds, periodic, scale_factor, align_target)
    lat = grid_out["lat"]
    lon = grid_out["lon"]
    
    np.save(os.path.join(save_dir, "lat.npy"), lat)
    np.save(os.path.join(save_dir, "lon.npy"), lon)
        
    nc2np(root_dir, src, variables, train_years, save_dir, "train", num_shards, frequency, regridder, periodic, scale_factor, align_target)
    nc2np(root_dir, src, variables, val_years, save_dir, "val", num_shards, frequency, regridder, periodic, scale_factor, align_target)
    nc2np(root_dir, src, variables, test_years, save_dir, "test", num_shards, frequency, regridder, periodic, scale_factor, align_target)
    
    
def open_era(root_dir, var):
    ps = glob.glob(os.path.join(root_dir, var, "*.zarr"))
    ds = xr.open_mfdataset(
            ps, combine="by_coords", parallel=True, engine="zarr"
            )
    return ds


def open_cmip(root_dir, var):
    ps = glob.glob(os.path.join(root_dir, var, "*.nc"))
    ds = xr.open_mfdataset(
            ps, combine="by_coords", parallel=True
        )
    return ds


def open_eobs(root_dir:str, var: str):
    ps = glob.glob(os.path.join(root_dir, f"{var}*.nc"))
    ds = xr.open_mfdataset(
            ps, combine="by_coords", parallel=True
        )
    # Clip source data with preset boundaries
    ds = ds.sel(latitude=slice(BOTTOM,TOP), longitude=slice(LEFT,RIGHT))
    return ds