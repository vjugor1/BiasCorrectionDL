import sys
sys.path.append('/app')
from src.climate_learn import LitModule
from src.climate_learn import download_weatherbench
from src.climate_learn import convert_nc2npz
from src.climate_learn import IterDataModule
from src.climate_learn.utils import visualize_at_index, visualize_mean_bias
from src.climate_learn.utils.gis import prepare_ynet_climatology, prepare_deepsd_elevation
from src.climate_learn import load_downscaling_module
from src.climate_learn.models.module import DiffusionLitModule, LitModule, DeepSDLitModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch
from era5_era5_dl import *
from hydra import compose, initialize
from omegaconf import OmegaConf
from torchvision import transforms
with initialize(version_base=None, config_path="../configs/train"):
    cfg = compose(config_name="era5-era5")


experiment_name = construct_experiment_name(cfg)
default_root_dir  = os.path.join(cfg.base_dir, experiment_name)

# Set the seed for reproducibility
pl.seed_everything(cfg.training.seed)

dm = setup_data_module(cfg)
module = setup_model(dm, cfg)
trainer = setup_trainer(cfg, default_root_dir )

elevation_list = prepare_deepsd_elevation(dm, path_to_elevation="/app/data/elevation.nc")
model = DeepSDLitModule.load_from_checkpoint(
        '/app/data/experiments/downscaling-ERA-ERA/deepsd_multi_none_0/logs/version_0/checkpoints/epoch_024.ckpt',
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

denorm = model.test_target_transforms[0]
in_graphic = visualize_at_index(
    model.to(device="cuda:1"),
    dm,
    in_transform=transforms.Normalize(torch.zeros(49), torch.ones(49)),
    out_transform=denorm,
    variable="2m_temperature",
    src="era5",
    index=0,
)