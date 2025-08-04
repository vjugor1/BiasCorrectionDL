from src.climate_learn import LitModule
from src.climate_learn.utils import visualize_at_index, visualize_mean_bias
from src.climate_learn.utils.gis import prepare_ynet_climatology, prepare_deepsd_elevation, prepare_dcgan_elevation
from src.climate_learn.models.module import DiffusionLitModule, LitModule, DeepSDLitModule, YnetLitModule, GANLitModule, ESRGANLitModule
from src.climate_learn.transforms import Mask, Denormalize

import pytorch_lightning as pl
import torch
import glob
import numpy as np
from hydra import compose, initialize
from torchvision import transforms


with initialize(version_base=None, config_path="configs/train"):
    cfg_train = compose(config_name="cmip6-cmip6")
with initialize(version_base=None, config_path="configs/inference"):
    cfg_inference = compose(config_name="cmip6-cmip6")

if cfg_inference.task=="era5-era5":
    from era5_era5_dl import *
    from era5_era5_baseline import load_model_baseline
    src="era5"
elif cfg_inference.task=="cmip6-cmip6":
    from cmip6_cmip6_dl import *
    from cmip6_cmip6_baseline import load_model_baseline
    src="cmip6"
elif cfg_inference.task=="era5-eobs":
    from era5_eobs_dl import *
    from era5_eobs_baseline import load_model_baseline
    src="era5-eobs"


def load_model(dm, arch):
    experiment_name = f"{arch}_multi_{cfg_inference[arch][0].upsampling}_{cfg_train.training.seed}"
    default_root_dir  = os.path.join(cfg_inference.path, experiment_name)  
    cfg_train.model.architecture = arch
    cfg_train.model.upsampling = cfg_inference[arch][0].upsampling
    path_to_elevation = cfg_inference.path_to_elevation
    module = setup_model(dm, cfg_train)
    trainer = setup_trainer(cfg_train, default_root_dir )

    i = cfg_inference.seeds.index(cfg_train.training.seed)
    ckpt = glob.glob(os.path.join(cfg_inference.path,
                                experiment_name,
                                f"logs/version_{cfg_inference[arch][1].versions[i]}/checkpoints/*.ckpt"))[-1]
    if arch=="ynet":
        normalized_clim = prepare_ynet_climatology(dm, path_to_elevation,
                                                out_vars=cfg_train.data.out_variables)
        module = YnetLitModule.load_from_checkpoint(
                ckpt,
                net=module.net,
                optimizer=module.optimizer,
                lr_scheduler=module.lr_scheduler,
                train_loss=module.train_loss,
                val_loss=module.val_loss,
                test_loss=module.test_loss,
                train_target_transform=module.train_target_transform,
                val_target_transforms=module.val_target_transforms,
                test_target_transforms=module.test_target_transforms,
                x_aux=normalized_clim,
            )

    elif arch == "deepsd":
        elevation_list = prepare_deepsd_elevation(dm, path_to_elevation)
        module = DeepSDLitModule.load_from_checkpoint(
                ckpt,
                net=module.net,
                optimizer=module.optimizer,
                lr_scheduler=module.lr_scheduler,
                train_loss=module.train_loss,
                val_loss=module.val_loss,
                test_loss=module.test_loss,
                train_target_transform=module.train_target_transform,
                val_target_transforms=module.val_target_transforms,
                test_target_transforms=module.test_target_transforms,
                elevation=elevation_list
        )

    elif arch=="dcgan":
        elevation = prepare_dcgan_elevation(dm, path_to_elevation)
        module = GANLitModule.load_from_checkpoint(
                ckpt,
                net=module.net,
                optimizer=module.optimizer,
                lr_scheduler=module.lr_scheduler,
                train_loss=module.train_loss,
                val_loss=module.val_loss,
                test_loss=module.test_loss,
                train_target_transform=module.train_target_transform,
                val_target_transforms=module.val_target_transforms,
                test_target_transforms=module.test_target_transforms,
                elevation=elevation
            )
    elif arch=="esrgan":
        module = ESRGANLitModule.load_from_checkpoint(
                ckpt,
                net=module.net,
                optimizer=module.optimizer,
                lr_scheduler=module.lr_scheduler,
                train_loss=module.train_loss,
                val_loss=module.val_loss,
                test_loss=module.test_loss,
                train_target_transform=module.train_target_transform,
                val_target_transforms=module.val_target_transforms,
                test_target_transforms=module.test_target_transforms,
    )
    elif arch=="diffusion":
        module = DiffusionLitModule.load_from_checkpoint(
                ckpt,
                net=module.net,
                optimizer=module.optimizer,
                lr_scheduler=module.lr_scheduler,
                train_loss=module.train_loss,
                val_loss=module.val_loss,
                test_loss=module.test_loss,
                train_target_transform=module.train_target_transform,
                val_target_transforms=module.val_target_transforms,
                test_target_transforms=module.test_target_transforms,
            )
    else:
        module = LitModule.load_from_checkpoint(
            ckpt,
            net=module.net,
            optimizer=module.optimizer,
            lr_scheduler=None,
            train_loss=None,
            val_loss=None,
            test_loss=module.test_loss,
            test_target_transforms=module.test_target_transforms,
        )

    denorm = module.test_target_transforms[0]
    return module, denorm


def make_plots():
    area = cfg_inference.area
    if area == "world":
        extent=None
    else:
        extent = [cfg_inference.areas[area]['left'],
                  cfg_inference.areas[area]['right'],
                  cfg_inference.areas[area]['bottom'],
                  cfg_inference.areas[area]['top']] 
    path_plots = os.path.join(cfg_inference.path, "plots", area)
    os.makedirs(path_plots, exist_ok=True)
    
    # The seed for reproducibility
    pl.seed_everything(cfg_train.training.seed)
    models = cfg_inference.models
    dm = setup_data_module(cfg_train)
    
    # Set up masking
    if cfg_inference.task=="era5-eobs":
        dm.mask = Mask(dm.out_mask)
        denorm = Denormalize(dm)
        dm.denorm_mask = lambda x: denorm(dm.mask(x))
    
    for var in cfg_inference.plot_variables:
        pred_min = bias_min = np.inf
        pred_max = bias_max = -np.inf
        
        # Define variable range among all models
        for arch in models:
            png_path = os.path.join(path_plots, f"{arch}_{cfg_train.training.seed}")
            if cfg_inference.task=="era5-eobs":
                module, _ = load_model(dm, arch)
            else:
                module, denorm = load_model(dm, arch)
            ppred, yy, img = visualize_at_index(
                module,
                dm,
                in_transform=transforms.Normalize(torch.zeros(dm.get_data_dims()[0][1]),
                                                  torch.ones(dm.get_data_dims()[0][1])),
                out_transform=denorm,
                variable=var,
                src=src,
                png_name=png_path,
                extent=extent,
                index=cfg_inference.time_stamp
            )
            pred_min = ppred.min() if ppred.min()<pred_min else pred_min
            pred_max = ppred.max() if ppred.max()>pred_max else pred_max
            pred_min = yy.min() if yy.min()<pred_min else pred_min
            pred_max = yy.max() if yy.max()>pred_max else pred_max
            
            bias = ppred - yy
            bias_min = np.nanmin(bias) if np.nanmin(bias)<bias_min else bias_min
            bias_max = np.nanmax(bias) if np.nanmax(bias)>bias_max else bias_max

        # Define variable range baseline. 
        # Comment this if you do not plot baseline
        dm, _, (bilinear, _) = load_model_baseline()
        denorm = bilinear.test_target_transforms[0]
        ppred, yy, img = visualize_at_index(
                bilinear,
                dm,
                in_transform=transforms.Normalize(torch.zeros(dm.get_data_dims()[0][1]),
                                                torch.ones(dm.get_data_dims()[0][1])),
                out_transform=denorm,
                variable=var,
                src=src,
                extent=extent,
                index=cfg_inference.time_stamp,
            )
        pred_min = ppred.min() if ppred.min()<pred_min else pred_min
        pred_max = ppred.max() if ppred.max()>pred_max else pred_max
        pred_min = yy.min() if yy.min()<pred_min else pred_min
        pred_max = yy.max() if yy.max()>pred_max else pred_max
        
        bias = ppred - yy
        bias_min = bias.min() if bias.min()<bias_min else bias_min
        bias_max = bias.max() if bias.max()>bias_max else bias_max
        
        # Make plots with known ranges
        var_range = [pred_min, pred_max, bias_min, bias_max]
        
        # Make plots baseline
        # Comment this if you do not plot baseline
        png_path = os.path.join(path_plots, "bilinear")
        _, _, _ = visualize_at_index(
                bilinear,
                dm,
                in_transform=transforms.Normalize(torch.zeros(dm.get_data_dims()[0][1]),
                                                torch.ones(dm.get_data_dims()[0][1])),
                out_transform=denorm,
                variable=var,
                src=src,
                png_name=png_path,
                extent=extent,
                index=cfg_inference.time_stamp,
                var_range = var_range
            )
        
        # Load general dm
        # Comment this if you do not plot baseline
        dm = setup_data_module(cfg_train)
        if cfg_inference.task=="era5-eobs":
            dm.mask = Mask(dm.out_mask)
            denorm = Denormalize(dm)
            dm.denorm_mask = lambda x: denorm(dm.mask(x))
        
        # Make plots other models
        for arch in models:
            png_path = os.path.join(path_plots, f"{arch}_{cfg_train.training.seed}")
            if cfg_inference.task=="era5-eobs":
                module, _ = load_model(dm, arch)
            else:
                module, denorm = load_model(dm, arch)
            _, _, _ = visualize_at_index(
                module,
                dm,
                in_transform=transforms.Normalize(torch.zeros(dm.get_data_dims()[0][1]),
                                                  torch.ones(dm.get_data_dims()[0][1])),
                out_transform=denorm,
                variable=var,
                src=src,
                png_name=png_path,
                extent=extent,
                index=cfg_inference.time_stamp,
                var_range = var_range
            )
      
if __name__ == "__main__":
    make_plots()