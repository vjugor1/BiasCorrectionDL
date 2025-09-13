import torch
from typing import Callable, List

@torch.no_grad()
def add_iid_gaussian(X: torch.Tensor, 
                    in_variables: List[str] = [
                        "air_temperature", "u_component_of_wind", "v_component_of_wind",
                        "precipitation", "pressure_sea_level", "specific_humidity",
                        "cloud_cover", "upward_heat_flux", "moisture_in_soil",
                        "geopotential_at_surface", "land_sea_mask", "latitude",
                        "standard_deviation_of_orography",
                        "standard_deviation_of_filtered_subgrid_orography",
                        "soil_type", "angle_of_sub_gridscale_orography"
                        ],
                    K: int = 10, 
                    sigma_per_channel: float = 0.06,
                    g=None):
    """
    X input data of size [B,C,H,W], 
    in_variables: lis of variables for every channel
    K:  len of creating ensemble
    sigma_per_channel: in normalized units for channel C, apply from range [0.01,0.03],
    """
    B, C, H, W = X.shape
    
    # Define variables excluded from noising
    no_noise_vars = [
        "geopotential_at_surface", "land_sea_mask", "latitude", 
        "standard_deviation_of_orography", "standard_deviation_of_filtered_subgrid_orography",
        "soil_type", "angle_of_sub_gridscale_orography", "orography",
        "precipitation"
    ]
    
    # Create mask for channels excluded from noising
    no_noise_mask = torch.zeros(C, dtype=torch.bool, device=X.device)
    for i, var_name in enumerate(in_variables):
        if var_name in no_noise_vars:
            no_noise_mask[i] = True
    
    # Apply noising
    sig = torch.full((C,), sigma_per_channel, device=X.device, dtype=X.dtype)
    sig[no_noise_mask] = 0.0
    sig = sig.view(1, -1, 1, 1)  # [1, C, 1, 1]
    
    gens = []
    for k in range(K):
        eps = torch.randn((B, C, H, W), generator=g, device=X.device, dtype=X.dtype)
        Xk = X + sig * eps
        gens.append(Xk)
    
    return torch.stack(gens, dim=1)  # [B, K, C, H, W]