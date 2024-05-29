import numpy as np
import xarray as xr
import xesmf

SCALE_FACTOR = 4
PATH_SRC = "/app/data/raw/era5_0.25deg/3H/constants_0.25.nc"
PATH_SAVE= "/app/data/raw/cmip6/3H/constants.nc"

def regrid(path_in, path_out, scale_factor):
    ds = xr.open_mfdataset(
                    path_in, combine="by_coords", parallel=True
                )

    lon_coarsen = ds["longitude"][::scale_factor].values
    lat_coarsen = ds["latitude"][::scale_factor].values
    grid_out = {'lon': lon_coarsen, 'lat': lat_coarsen}
    
    regridder = xesmf.Regridder(ds,
                                grid_out,
                                "bilinear",
                                periodic=True
                                )
    ds_regrid = regridder(ds, keep_attrs=True)
    ds_regrid["latitude"] = xr.DataArray(
                        data = np.array([ds_regrid['lat']]*len(ds_regrid["lon"])).T,
                        coords=ds_regrid.coords
                        )
    if len(ds_regrid.lat)%2!=0:
        ds_regrid = ds_regrid.isel(lat=slice(0, len(ds_regrid.lat)//2*2))
    ds_regrid.to_netcdf(path_out)
        
        
if __name__=="__main__":
    regrid(PATH_SRC, PATH_SAVE, SCALE_FACTOR)
