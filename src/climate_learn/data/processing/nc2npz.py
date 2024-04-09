# Standard library
import glob
import os

# Third party
import numpy as np
import xarray as xr
import netCDF4 as nc
from tqdm import tqdm
import xesmf

# Local application
from .era5_constants import (
    DEFAULT_PRESSURE_LEVELS_ERA,
    NAME_TO_VAR_ERA,
    VAR_TO_NAME_ERA,
    CONSTANTS_ERA,
)

from .cmip6_constants import (
    DEFAULT_PRESSURE_LEVELS_CMIP,
    NAME_TO_VAR_CMIP,
    VAR_TO_NAME_CMIP,
    CONSTANTS_CMIP,
)


def nc2np(path,
          src,
          variables,
          years,
          save_dir,
          partition,
          num_shards_per_year,
          frequency,
          regridder):
    if src=="era5":
        DEFAULT_PRESSURE_LEVELS=DEFAULT_PRESSURE_LEVELS_ERA
        NAME_TO_VAR=NAME_TO_VAR_ERA
        VAR_TO_NAME=VAR_TO_NAME_ERA
        CONSTANTS=CONSTANTS_ERA
    elif src=="cmip6":
        DEFAULT_PRESSURE_LEVELS=DEFAULT_PRESSURE_LEVELS_CMIP
        NAME_TO_VAR=NAME_TO_VAR_CMIP
        VAR_TO_NAME=VAR_TO_NAME_CMIP
        CONSTANTS=CONSTANTS_CMIP

    assert frequency in ["3H", "D"]
    assert src in ["era5", "cmip6", "eobs"]
    if frequency=="3H":
        OBJ_PER_YEAR = 365*8 #(3hr period yields 8 obj per day)
    elif frequency=="D":
        OBJ_PER_YEAR = 365*24

    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)

    if partition == "train":
        normalize_mean = {}
        normalize_std = {}
    climatology = {}

    constants_path = os.path.join(path, "constants.nc")
    constants_are_downloaded = os.path.isfile(constants_path)

    if constants_are_downloaded:
        constants = xr.open_mfsrc(
            constants_path, combine="by_coords", parallel=True
        )
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
            code = NAME_TO_VAR[var]

            if src=="era5":
                ps = glob.glob(os.path.join(path, var, f"*{year}.zarr"))
                ds = xr.open_mfdataset(
                    ps, combine="by_coords", parallel=True, engine="zarr"
                )
                ds = ds.rename({'longitude': 'lon','latitude': 'lat'})
                ds.transpose("time", "lat", "lon", ...)
                # ds.coords["lon"] = (ds.coords["lon"] + 180) % 360 - 180
                
            elif src=="cmip6":
                ps = glob.glob(os.path.join(path, var, "*.nc"))
                ds = xr.open_mfdataset(
                    ps, combine="by_coords", parallel=True
                    )
                ds = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
                # ds.coords["lon"] = (ds.coords["lon"] + 180) % 360 - 180
                # Align in latitude direction with era
                # ds  = ds.sortby('lat', ascending=False)
            
            if regridder!= None:
                ds = regridder(ds, keep_attrs=True)
                
            # cut last value if len(latitude) is odd == make it even
            if len(ds.lat)%2!=0:
                ds = ds.isel(lat=slice(0, len(ds.lat)//2*2))

            if len(ds[code].shape) == 3:  # surface level variables
                ds[code] = ds[code].expand_dims("val", axis=1)
                # remove the last 24 hours if this year has 366 days
                if code == "tp":  # accumulate 6 hours and log transform
                    tp = ds[code].to_numpy()
                    tp_cum_6hrs = np.cumsum(tp, axis=0)
                    tp_cum_6hrs[6:] = tp_cum_6hrs[6:] - tp_cum_6hrs[:-6]
                    eps = 0.001
                    tp_cum_6hrs = np.log(eps + tp_cum_6hrs) - np.log(eps)
                    np_vars[var] = tp_cum_6hrs[:OBJ_PER_YEAR]
                else:
                    np_vars[var] = ds[code].to_numpy()[:OBJ_PER_YEAR]
                    
                if partition == "train":
                    # compute mean and std of each var in each year
                    var_mean_yearly = np_vars[var].mean(axis=(0, 2, 3))
                    var_std_yearly = np_vars[var].std(axis=(0, 2, 3))
                    if var not in normalize_mean:
                        normalize_mean[var] = [var_mean_yearly]
                        normalize_std[var] = [var_std_yearly]
                    else:
                        normalize_mean[var].append(var_mean_yearly)
                        normalize_std[var].append(var_std_yearly)

                clim_yearly = np_vars[var].mean(axis=0)
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
        num_hrs_per_shard = OBJ_PER_YEAR // num_shards_per_year
        for shard_id in range(num_shards_per_year):
            start_id = shard_id * num_hrs_per_shard
            end_id = start_id + num_hrs_per_shard
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
                if var == "total_precipitation":
                    normalize_mean[var] = np.zeros_like(normalize_mean[var])
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


def regrid(ds_in, align_target):
    ps1 = glob.glob(os.path.join(align_target, "*"))
    ps2 = glob.glob(os.path.join(align_target, ps1[0], "*.zarr"))
    ds = xr.open_mfdataset(
                    ps2, combine="by_coords", parallel=True, engine="zarr"
                )
    lon_coarsen = ds["longitude"][::4].values
    lat_coarsen = ds["latitude"][::4].values
    grid_out = {'lon': lon_coarsen, 'lat': lat_coarsen}
    
    regridder = xesmf.Regridder(ds_in,
                                grid_out,
                                "bilinear",
                                periodic=True
                                )
    return regridder, grid_out


def convert_nc2npz(
    root_dir,
    save_dir,
    src,
    variables,
    start_train_year,
    start_val_year,
    start_test_year,
    end_year,
    num_shards,
    frequency,
    align_target
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
    
    if src=="era5":
        ps = glob.glob(os.path.join(root_dir, variables[0], "*.zarr"))
        ds = xr.open_mfdataset(
            ps, combine="by_coords", parallel=True, engine="zarr"
        )
        ds = ds.rename({'longitude': 'lon','latitude': 'lat'})
        
    elif src=="cmip6":
        ps = glob.glob(os.path.join(root_dir, variables[0], "*.nc"))
        ds = xr.open_mfdataset(
            ps, combine="by_coords", parallel=True
            )

    # create regridder and save lat/lon data
    if align_target!= None: 
        regridder, grid_out = regrid(ds, align_target)
        lat = grid_out["lat"]
        lon = grid_out["lon"]
    else:
        regridder=None
        lat = np.array(ds["lat"])
        lon = np.array(ds["lon"])
    
    # cut last value if len(latitude) is odd == make it even
    if len(lat)%2!=0:
        lat = lat[:-1]
    np.save(os.path.join(save_dir, "lat.npy"), lat)
    np.save(os.path.join(save_dir, "lon.npy"), lon)
        
    nc2np(root_dir, src, variables, train_years, save_dir, "train", num_shards, frequency, regridder)
    nc2np(root_dir, src, variables, val_years, save_dir, "val", num_shards, frequency, regridder)
    nc2np(root_dir, src, variables, test_years, save_dir, "test", num_shards, frequency, regridder)