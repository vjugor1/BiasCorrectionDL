import numpy as np
import xarray as xr
import xesmf

SCALE_FACTOR = 4
PATH_SRC = "/app/data/raw/cmip6-era5/era5_0.25/constants.nc"
# PATH_SAVE= "/app/data/raw/era5-eobs/era5_0.25_D/constants.nc"
PATH_SAVE= "/app/data/raw/constants.nc"

def regrid(path_in, path_out, scale_factor, periodic=True):
    ds = xr.open_mfdataset(
                    path_in, combine="by_coords", parallel=True
                )
    if len(ds["latitude"])/scale_factor%2!=0:
            n_cells_lat=int((len(ds["latitude"])-1)/scale_factor)
    if periodic==True:
        n_cells_lon=int(len(ds["longitude"])/scale_factor)
    else:
        n_cells_lon=int((len(ds["longitude"])-1)/scale_factor)
        
    lon_new = np.linspace(
        np.min(ds["longitude"].values),
        np.max(ds["longitude"].values),
        n_cells_lon)
    lat_new = np.linspace(
        np.min(ds["latitude"].values),
        np.max(ds["latitude"].values),
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
    regrid(PATH_SRC, PATH_SAVE, SCALE_FACTOR)
