import xarray as xr
import numpy as np
import torch
import os
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from affine import Affine


def prepare_ynet_climatology(data_module, path_to_elevation, out_vars):
    """Prepares the YNet climatology data by normalizing and adding elevation data."""
    if path_to_elevation is None:
        raise TypeError("The path to elevation dataset is None.")
    
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
    with xr.open_dataset(path_to_elevation) as elevation_ds:
        elevation_ds = (
            elevation_ds.assign_coords(X=(elevation_ds.X % 360))
            .sortby("X")
            .rename({"X": "lon", "Y": "lat"})
        )
        elevation_ds["topo"].values = (
            np.log1p(np.nan_to_num(elevation_ds["topo"].values)) / 9
        )

    # Calculate the high-resolution transform
    high_res_transform = _calculate_transform(elevation_ds)
    
    target_lat, target_lon = data_module.get_lat_lon()

    # Ensure latitude array is in descending order
    target_lat = target_lat[::-1] if target_lat[0] < target_lat[-1] else target_lat

    target_shape = (len(target_lat), len(target_lon))
    target_transform = from_bounds(
        target_lon.min(), target_lat.min(),
        target_lon.max(), target_lat.max(),
        len(target_lon), len(target_lat),
    )

    # Reproject the data
    reprojected_data = _reproject_elevation(elevation_ds["topo"].values, target_shape, high_res_transform, target_transform)

    # Create the new reprojected dataset
    new_elevation_ds = xr.Dataset(
        {"topo": (["lat", "lon"], reprojected_data)},
        coords={"lat": target_lat, "lon": target_lon},
    )

    new_elevation_tensor = torch.from_numpy(new_elevation_ds["topo"].values).unsqueeze(0)
    normalized_clim = torch.cat((normalized_clim, new_elevation_tensor), dim=0)

    return normalized_clim


def get_climatology(data_module, split):
    """Retrieves the climatology data for the given split."""
    clim = data_module.get_climatology(split=split)
    if clim is None:
        raise RuntimeError("Climatology has not yet been set.")
    if isinstance(clim, dict):
        clim = torch.stack(tuple(clim.values()))
    return clim


def _calculate_transform(elevation_ds):
    """Calculates the affine transform for the given dataset."""
    lon_min, lon_max = elevation_ds.lon.min().item(), elevation_ds.lon.max().item()
    lat_min, lat_max = elevation_ds.lat.min().item(), elevation_ds.lat.max().item()
    res_lon = (lon_max - lon_min) / (elevation_ds.dims["lon"] - 1)
    res_lat = (lat_max - lat_min) / (elevation_ds.dims["lat"] - 1)
    return Affine.translation(
        lon_min - res_lon / 2, lat_min - res_lat / 2
    ) * Affine.scale(res_lon, res_lat)


def _reproject_elevation(source_data, target_shape, src_transform, dst_transform):
    """Reprojects the source elevation data to the target shape."""
    reprojected_data = np.empty(target_shape, dtype=np.float32)
    reproject(
        source=source_data,
        destination=reprojected_data,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.bilinear,
    )
    return reprojected_data


def prepare_deepsd_elevation(data_module, path_to_elevation):
    """Prepares the DeepSD elevation data at various scales."""
    if path_to_elevation is None:
        raise TypeError("The path to elevation dataset is None.")
    
    if not os.path.isfile(path_to_elevation):
        raise FileNotFoundError(f"The file at {path_to_elevation} does not exist.")
    
    in_shape, out_shape = data_module.get_data_dims()
    _, _, in_width = in_shape[1:]
    _, out_height, out_width = out_shape[1:]

    if out_height % 2 != 0 or out_width % 2 != 0:
        raise ValueError("Output height and width must be divisible by 2.")
    
    scale_factor = out_width / in_width
    if scale_factor != int(scale_factor):
        raise ValueError("Scale factor must be an integer.")
    
    scale_factor = int(scale_factor)
    if scale_factor % 2 != 0 or (scale_factor & (scale_factor - 1)) != 0:
        raise ValueError("Scale factor must be a power of 2.")

    # Load and preprocess the high-resolution dataset
    with xr.open_dataset(path_to_elevation) as elevation_ds:
        elevation_ds = (
            elevation_ds.assign_coords(X=(elevation_ds.X % 360))
            .sortby("X")
            .rename({"X": "lon", "Y": "lat"})
        )
        elevation_ds["topo"].values = (
            np.log1p(np.nan_to_num(elevation_ds["topo"].values)) / 9
        )

    # Initial setup for reprojection
    target_lat, target_lon = data_module.get_lat_lon()
    target_lat = np.flip(target_lat) if target_lat[0] < target_lat[-1] else target_lat

    elevation_list = []
    current_shape = (len(target_lat), len(target_lon))
    current_transform = from_bounds(
        target_lon.min(), target_lat.min(),
        target_lon.max(), target_lat.max(),
        current_shape[1], current_shape[0],
    )

    # Reproject elevation data at different scales
    for _ in range(int(np.log2(scale_factor))):
        reprojected_data = _reproject_elevation(
            elevation_ds["topo"].values, current_shape,
            _calculate_transform(elevation_ds), current_transform
        )
        elevation_tensor = torch.from_numpy(reprojected_data).unsqueeze(0)
        elevation_list.append(elevation_tensor)

        # Update the target shape and transform for the next scale
        current_shape = (current_shape[0] // 2, current_shape[1] // 2)
        current_transform = from_bounds(
            target_lon.min(), target_lat.min(),
            target_lon.max(), target_lat.max(),
            current_shape[1], current_shape[0],
        )   
    return elevation_list[::-1]

def prepare_dcgan_elevation(data_module, path_to_elevation):
    """Prepares the DCGAN elevation data at the output size."""
    
    if path_to_elevation is None:
        raise TypeError("The path to elevation dataset is None.")
    
    if not os.path.isfile(path_to_elevation):
        raise FileNotFoundError(f"The file at {path_to_elevation} does not exist.")

    # Load and preprocess the high-resolution dataset
    with xr.open_dataset(path_to_elevation) as elevation_ds:
        elevation_ds = (
            elevation_ds.assign_coords(X=(elevation_ds.X % 360))
            .sortby("X")
            .rename({"X": "lon", "Y": "lat"})
        )
        elevation_ds["topo"].values = (
            np.log1p(np.nan_to_num(elevation_ds["topo"].values)) / 9
        )

    # Calculate the high-resolution transform
    high_res_transform = _calculate_transform(elevation_ds)
    
    target_lat, target_lon = data_module.get_lat_lon()

    # Ensure latitude array is in descending order
    target_lat = target_lat[::-1] if target_lat[0] < target_lat[-1] else target_lat

    target_shape = (len(target_lat), len(target_lon))
    target_transform = from_bounds(
        target_lon.min(), target_lat.min(),
        target_lon.max(), target_lat.max(),
        len(target_lon), len(target_lat),
    )
    
    # Reproject the data
    reprojected_data = _reproject_elevation(elevation_ds["topo"].values, target_shape, high_res_transform, target_transform)
    
    # Create the new reprojected dataset
    new_elevation_ds = xr.Dataset(
        {"topo": (["lat", "lon"], reprojected_data)},
        coords={"lat": target_lat, "lon": target_lon},
    )

    new_elevation_tensor = torch.from_numpy(new_elevation_ds["topo"].values).unsqueeze(0)
    
    return new_elevation_tensor


def prepare_elevation(path_to_elevation, target_path):
    """Prepares elevation to concatenate with LR data"""
    
    if path_to_elevation is None:
        raise TypeError("The path to elevation dataset is None.")
    
    if not os.path.isfile(path_to_elevation):
        raise FileNotFoundError(f"The file at {path_to_elevation} does not exist.")

    if "constant" not in target_path:
        raise FileNotFoundError(f"The file at {target_path} must be constants.nc type")
        
    # Load and preprocess the high-resolution dataset
    with xr.open_dataset(path_to_elevation) as elevation_ds:
        elevation_ds = (
            elevation_ds.assign_coords(X=(elevation_ds.X % 360))
            .sortby("X")
            .rename({"X": "lon", "Y": "lat"})
        )
        elevation_ds["topo"].values = (
            np.log1p(np.nan_to_num(elevation_ds["topo"].values)) / 9
        )

    # Calculate the high-resolution transform
    high_res_transform = _calculate_transform(elevation_ds)
    
    with xr.open_dataset(target_path) as ds_target:
        lat_axis = [k for k in list(ds_target.dims) if 'lat' in k][0]
        lon_axis = [k for k in list(ds_target.dims) if 'lon' in k][0]

    if len(ds_target[lat_axis])%2!=0:
            n_cells_lat=(len(ds_target[lat_axis])-1)
    else:
        n_cells_lat=len(ds_target[lat_axis])
    n_cells_lon=len(ds_target[lon_axis])
    lon_new = np.linspace(
        np.min(ds_target[lon_axis].values),
        np.max(ds_target[lon_axis].values),
        int(n_cells_lon))
    lat_new = np.linspace(
        np.max(ds_target[lat_axis].values),
        np.min(ds_target[lat_axis].values),
        int(n_cells_lat))
    
    target_lat, target_lon = lat_new, lon_new

    target_shape = (n_cells_lat, n_cells_lon)
    target_transform = from_bounds(
        target_lon.min(), target_lat.min(),
        target_lon.max(), target_lat.max(),
        len(target_lon), len(target_lat),
    )
    
    # Reproject the data
    reprojected_data = _reproject_elevation(elevation_ds["topo"].values, target_shape, high_res_transform, target_transform)
    ds_target["elevation"] = (['latitude', 'longitude'],  reprojected_data)
    ds_target.to_netcdf(os.path.join("/".join(target_path.split("/")[:-1]), "temp.nc"), mode='w')  
    
    ds_target = xr.open_dataset(os.path.join("/".join(target_path.split("/")[:-1]), "temp.nc"))
    ds_target.to_netcdf(target_path, mode = 'w')