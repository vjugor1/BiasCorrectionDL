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
                    sigma_per_channel: float = 0.02,
                    g=None):
    """
    X input data of size [B,C,H,W], 
    in_variables: список названий переменных для каждого канала
    K: размер ансамбля
    sigma_per_channel: стандартное отклонение шума для каждого канала
    """
    B, C, H, W = X.shape
    
    # Определяем переменные, которые НЕ нужно зашумлять
    no_noise_vars = [
        "geopotential_at_surface", "land_sea_mask", "latitude", 
        "standard_deviation_of_orography", "standard_deviation_of_filtered_subgrid_orography",
        "soil_type", "angle_of_sub_gridscale_orography", "orography",
        "precipitation"
    ]
    
    # Создаем маску для каналов, которые не нужно зашумлять
    no_noise_mask = torch.zeros(C, dtype=torch.bool, device=X.device)
    for i, var_name in enumerate(in_variables):
        if var_name in no_noise_vars:
            no_noise_mask[i] = True
    
    # Применяем sigma только к каналам, которые нужно зашумлять
    sig = torch.full((C,), sigma_per_channel, device=X.device, dtype=X.dtype)
    sig[no_noise_mask] = 0.0  # Зануляем sigma для защищенных переменных
    sig = sig.view(1, -1, 1, 1)  # [1, C, 1, 1]
    
    gens = []
    for k in range(K):
        eps = torch.randn((B, C, H, W), generator=g, device=X.device, dtype=X.dtype)
        Xk = X + sig * eps
        gens.append(Xk)
    
    return torch.stack(gens, dim=1)  # [B, K, C, H, W]