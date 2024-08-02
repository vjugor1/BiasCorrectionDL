from src.climate_learn import LitModule
from src.climate_learn import IterDataModule
from src.climate_learn.utils import visualize_at_index, visualize_mean_bias
from src.climate_learn.utils.gis import prepare_ynet_climatology, prepare_deepsd_elevation, prepare_dcgan_elevation
from src.climate_learn import load_downscaling_module
from src.climate_learn.models.module import DiffusionLitModule, LitModule, DeepSDLitModule, YnetLitModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch
import glob
from era5_era5_dl import *
from hydra import compose, initialize
from torchvision import transforms
from matplotlib import pyplot as plt

with initialize(version_base=None, config_path="configs/train"):
    cfg_train = compose(config_name="era5-era5")
with initialize(version_base=None, config_path="configs/inference"):
    cfg_inference = compose(config_name="era5_era5")

def make_plots():
    path_to_elevation = cfg_inference.path_to_elevation
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
    
    # Set the seed for reproducibility
    pl.seed_everything(cfg_train.training.seed)
    models = cfg_inference.models
    dm = setup_data_module(cfg_train)
    
    for arch in models:
        experiment_name = f"{arch}_multi_{cfg_inference[arch][0].upsampling}_{cfg_train.training.seed}"
        default_root_dir  = os.path.join(cfg_inference.path, experiment_name)
        png_path = os.path.join(path_plots, arch)
        
        cfg_train.model.architecture = arch
        cfg_train.model.upsampling = cfg_inference[arch][0].upsampling
        print(cfg_train)
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
        for variable in cfg_inference.plot_variables:
            in_graphic = visualize_at_index(
                module.to(device="cuda:1"),
                dm,
                in_transform=transforms.Normalize(torch.zeros(49), torch.ones(49)),
                out_transform=denorm,
                variable="2m_temperature",
                src="era5",
                png_name=png_path,
                extent=extent,
                index=cfg_inference.time_stamp
            )

if __name__ == "__main__":
    make_plots()