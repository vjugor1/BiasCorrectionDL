import xarray as xr
import numpy as np
import torch
import os
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from affine import Affine


def prepare_ynet_climatology(data_module, path_to_elevation, out_vars):
    # Check if path_to_elevation is None
    if path_to_elevation is None:
        raise TypeError("The path to elevation dataset is None.")
    
    # Check if the file exists
    if not os.path.isfile(path_to_elevation):
        raise FileNotFoundError(f"The file at {path_to_elevation} does not exist.")
    
    # Get climatology and normalize it
    clim = get_climatology(data_module, "train")
    norm = data_module.get_out_transforms()
    normalized_clim = clim.clone()
    for i, var in enumerate(out_vars):
        mean = norm[var].mean
        std = norm[var].std
        normalized_clim[i, :, :] = (clim[i, :, :] - mean) / std

    # Load and preprocess the high-resolution dataset
    elevation_ds = xr.open_dataset(path_to_elevation)
    elevation_ds = (
        elevation_ds.assign_coords(X=(elevation_ds.X % 360))
        .sortby("X")
        .rename({"X": "lon", "Y": "lat"})
    )
    elevation_ds["topo"].values = (
        np.log1p(np.nan_to_num(elevation_ds["topo"].values)) / 9
    )

    # Calculate the high-resolution transform
    lon_min, lon_max = elevation_ds.lon.min().item(), elevation_ds.lon.max().item()
    lat_min, lat_max = elevation_ds.lat.min().item(), elevation_ds.lat.max().item()
    res_lon = (lon_max - lon_min) / (elevation_ds.dims["lon"] - 1)
    res_lat = (lat_max - lat_min) / (elevation_ds.dims["lat"] - 1)
    high_res_transform = Affine.translation(
        lon_min - res_lon / 2, lat_min - res_lat / 2
    ) * Affine.scale(res_lon, res_lat)

    # Define target latitude and longitude arrays (example arrays)
    target_lat, target_lon = data_module.get_lat_lon()

    # Ensure latitude array is in descending order
    if target_lat[0] < target_lat[-1]:
        target_lat = target_lat[::-1]

    # Create the target transform and shape
    target_shape = (len(target_lat), len(target_lon))
    target_transform = from_bounds(
        target_lon.min(),
        target_lat.min(),
        target_lon.max(),
        target_lat.max(),
        len(target_lon),
        len(target_lat),
    )

    # Reproject the data
    reprojected_data = np.empty(target_shape, dtype=np.float32)
    reproject(
        source=elevation_ds["topo"].values,
        destination=reprojected_data,
        src_transform=high_res_transform,
        src_crs="EPSG:4326",
        dst_transform=target_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.bilinear,
    )

    # Create the new reprojected dataset
    new_elevation_ds = xr.Dataset(
        {"topo": (["lat", "lon"], reprojected_data)},
        coords={"lat": target_lat, "lon": target_lon},
    )

    new_elevation_tensor = torch.from_numpy(new_elevation_ds["topo"].values).unsqueeze(
        0
    )

    normalized_clim = torch.cat((normalized_clim, new_elevation_tensor), dim=0)

    return normalized_clim


def get_climatology(data_module, split):
    clim = data_module.get_climatology(split=split)
    if clim is None:
        raise RuntimeError("Climatology has not yet been set.")
    # Hotfix to work with dict style data
    if isinstance(clim, dict):
        clim = torch.stack(tuple(clim.values()))
    return clim