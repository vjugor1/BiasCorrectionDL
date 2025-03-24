import numpy as np
import xarray as xr
import xesmf
import yaml

SCALE_FACTOR = 1
PATH_SRC = "/app/data/raw/constants.nc"
PATH_TARGET = "/app/data/raw/e-obs/ensemble_mean/020_grid/1950-2023/t*.nc" #useone of temperat files in eobs, for example t*.nc
PATH_SAVE= "/app/data/raw/e-obs/ensemble_mean/020_grid/constants.nc"

# Get eobs bounds
with open("/app/configs/load/era.yaml", 'r') as file:
    cfg = yaml.safe_load(file)
TOP = cfg["bounds"]["top"]
BOTTOM = cfg["bounds"]["bottom"]
LEFT = cfg["bounds"]["left"]
RIGHT = cfg["bounds"]["right"]


def regrid(path_in,
           path_out,
           scale_factor=1,
           path_target = None,
           periodic=True):
    
    ds = xr.open_mfdataset(
                    path_in, combine="by_coords", parallel=True
                )

    if not path_target:
        ds_target=ds_in
    else:
        ds_target=xr.open_mfdataset(
                    path_target, combine="by_coords", parallel=True
                )
        
    if "e-obs" in path_out:
        # Clip source data with preset boundaries
        ds = ds.sel(latitude=slice(TOP, BOTTOM), longitude=slice(LEFT,RIGHT))
        ds_target = ds_target.sel(latitude=slice(BOTTOM,TOP), longitude=slice(LEFT,RIGHT))
        print(ds)
        print(ds_target)
        periodic = False
        
    lat_axis = [k for k in list(ds_target.dims) if 'lat' in k][0]
    lon_axis = [k for k in list(ds_target.dims) if 'lon' in k][0]
    if len(ds_target[lat_axis])/scale_factor%2!=0:
        n_cells_lat=int((len(ds_target[lat_axis])-1)/scale_factor)
    else:
        n_cells_lat=int(len(ds_target[lat_axis])/scale_factor)
    if periodic==True:
        n_cells_lon=int(len(ds_target[lon_axis])/scale_factor)
    else:
        n_cells_lon=int((len(ds_target[lon_axis])-1)/scale_factor)
        
    lon_new = np.linspace(
        np.min(ds_target[lon_axis].values),
        np.max(ds_target[lon_axis].values),
        n_cells_lon)
    lat_new = np.linspace(
        np.min(ds_target[lat_axis].values),
        np.max(ds_target[lat_axis].values),
        n_cells_lat)
    
    grid_out = {'lon': lon_new, 'lat': lat_new}
    print(grid_out)
    
    regridder = xesmf.Regridder(ds,
                                grid_out,
                                "bilinear",
                                periodic=periodic
                                )
    ds_regrid = regridder(ds, keep_attrs=True)
    ds_regrid["lat_grid"] = xr.DataArray(
                        data = np.array([ds_regrid['lat']]*len(ds_regrid["lon"])).T,
                        coords=ds_regrid.coords
                        )
    ds_regrid = ds_regrid.rename({'lon': 'longitude', 'lat': 'latitude'})
    print(ds_regrid.dims, ds_regrid)
    ds_regrid.to_netcdf(path_out)
        
        
if __name__=="__main__":
    regrid(PATH_SRC,
           PATH_SAVE,
           SCALE_FACTOR,
           PATH_TARGET)
