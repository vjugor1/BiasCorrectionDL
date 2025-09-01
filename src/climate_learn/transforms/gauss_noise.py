import torch


# def standardize(X, mean, std):
#     # X: [b,c,x,y]; mean,std: [c] или [1,c,1,1]
#     return (X - mean.view(1, -1, 1, 1)) / (std.view(1, -1, 1, 1) + 1e-8)


# def destandardize(Z, mean, std):
#     return Z * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


@torch.no_grad()
def add_iid_gaussian(X: torch.Tensor, 
                    # mean: float, 
                    # std: float,
                    K: int=10, 
                    sigma_per_channel: float=0.01, 
                    g=None):
    """
    X input data of size [B,C,H,W], 
    mean mean along variable (for channel C)
    std
    K len of creating ensemble
    sigma_per_channel: in normalized units for channel C, apply from range [0.01,0.03],
    
    """
    B, C, H, W = X.shape
    # Xn = standardize(X, mean, std)
    # sig = sigma_per_channel.view(1, -1, 1, 1)
    sig = torch.tensor(sigma_per_channel).unsqueeze(0).unsqueeze(1).unsqueeze(1).to(X.device)
    gens = []
    for k in range(K):
        eps = torch.randn((B, C, H, W), generator=g, device=X.device, dtype=X.dtype)
        Xk = X + sig * eps
        gens.append(Xk) #destandardize(Xk, mean, std))
    return torch.stack(gens, dim=1)  # [B, K, C, H, W]