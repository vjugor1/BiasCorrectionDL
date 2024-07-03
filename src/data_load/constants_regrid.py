import numpy as np
import xarray as xr
import xesmf

SCALE_FACTOR = 1
PATH_SRC = "/app/data/raw/cmip6-era5/era5_0.25/constants.nc"
PATH_TARGET = "/app/data/raw/cmip6-cmip6/LR/cloud_cover/*.nc"
PATH_SAVE= "/app/data/raw/constants.nc"

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
    
    regridder = xesmf.Regridder(ds,
                                grid_out,
                                "bilinear",
                                periodic=True
                                )
    ds_regrid = regridder(ds, keep_attrs=True)
    ds_regrid["lat_grid"] = xr.DataArray(
                        data = np.array([ds_regrid['lat']]*len(ds_regrid["lon"])).T,
                        coords=ds_regrid.coords
                        )
    ds_regrid = ds_regrid.rename({'lon': 'longitude', 'lat': 'latitude'})
    ds_regrid.to_netcdf(path_out)
        
        
if __name__=="__main__":
    regrid(PATH_SRC,
           PATH_SAVE,
           SCALE_FACTOR,
           PATH_TARGET)
