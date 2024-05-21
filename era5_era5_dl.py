# Standard library
from argparse import ArgumentParser

import pytorch_lightning as pl
# Third party
import torch
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, RichModelSummary,
                                         RichProgressBar)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from src.climate_learn import (IterDataModule, LitModule,
                               load_downscaling_module)
from src.climate_learn.data.processing.era5_constants import (
    DEFAULT_PRESSURE_LEVELS, PRESSURE_LEVEL_VARS)

# torch.set_float32_matmul_precision("medium")

parser = ArgumentParser()
parser.add_argument(
    "--default_root_dir",
    type=str,
    default="/app/data/experiments/downscaling-ERA-ERA/resnet_t2mgt_bilinear",
)
parser.add_argument(
    "--era5_low_res_dir",
    type=str,
    default="/app/data/ClimateLearn/processed/weatherbench/era5/5.625deg",
)
parser.add_argument(
    "--era5_high_res_dir",
    type=str,
    default="/app/data/ClimateLearn/processed/weatherbench/era5/2.8125deg",
)
parser.add_argument(
    "--out_variables",
    choices=["2m_temperature", "geopotential_500", "temperature_850"],
    help="The variable to predict.",
    default=["2m_temperature", "geopotential_500", "temperature_850"],
)
parser.add_argument(
    "--architecture",
    type=str,
    choices=["resnet", "unet", "vit", "samvit", "ynet"],
    default="resnet",
)
parser.add_argument(
    "--upsampling",
    type=str,
    choices=["bilinear", "bicubic", "unet_upsampling", "unet_upsampling_bilinear", "none"],
    default="bilinear",
)
parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=200)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--gpu", type=list, default=[0])
parser.add_argument("--checkpoint", default=None)
args = parser.parse_args()


# Set up data
variables = [
    "land_sea_mask",
    "orography",
    "lattitude",
    "toa_incident_solar_radiation",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "geopotential",
    "temperature",
    "relative_humidity",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
]
in_vars = []
for var in variables:
    if var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            in_vars.append(var + "_" + str(level))
    else:
        in_vars.append(var)

dm = IterDataModule(
    task="downscaling",
    inp_root_dir=args.era5_low_res_dir,
    out_root_dir=args.era5_high_res_dir,
    in_vars=in_vars,
    out_vars=args.out_variables,
    subsample=1,
    batch_size=64,
    buffer_size=10000,
    num_workers=4,
)
dm.setup()

# Set up deep learning model
model = load_downscaling_module(
    data_module=dm,
    architecture=args.architecture,
    upsampling=args.upsampling,
    # model_kwargs={"neck_chans": 256}, # {"mid_attn": True}
    optim_kwargs={"lr": 1e-5},
    sched="linear-warmup-cosine-annealing",
    sched_kwargs={"warmup_epochs": 5, "max_epochs": 50},
    train_loss="mse",
    val_loss=["rmse", "pearson", "mean_bias", "mse"],
    test_loss=["rmse", "pearson", "mean_bias", "mse"],
    train_target_transform=None,
    val_target_transform=["denormalize", "denormalize", "denormalize", None],
    test_target_transform=["denormalize", "denormalize", "denormalize", None],
)

# Setup trainer
pl.seed_everything(414)
logger = TensorBoardLogger(save_dir=f"{args.default_root_dir}/logs")
early_stopping = "val/mse:aggregate"
callbacks = [
    RichProgressBar(),
    RichModelSummary(max_depth=args.summary_depth),
    EarlyStopping(
        monitor=early_stopping,
        min_delta=1e-4,
        patience=args.patience,
        verbose=True,
        mode="min",
    ),
    ModelCheckpoint(
        dirpath=f"{args.default_root_dir}/checkpoints",
        monitor=early_stopping,
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
    ),
    LearningRateMonitor(logging_interval="epoch"),
]
trainer = pl.Trainer(
    enable_progress_bar=True,
    logger=logger,
    callbacks=callbacks,
    default_root_dir=args.default_root_dir,
    accelerator="gpu",
    devices=args.gpu,
    max_epochs=args.max_epochs,
    # strategy="ddp",
    precision="16",
)

# Train and evaluate model from scratch
if args.checkpoint is None:
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")
# Evaluate saved model checkpoint
else:
    model = LitModule.load_from_checkpoint(
        args.checkpoint,
        net=model.net,
        optimizer=model.optimizer,
        lr_scheduler=None,
        train_loss=None,
        val_loss=None,
        test_loss=model.test_loss,
        test_target_tranfsorms=model.test_target_transforms,
    )
    trainer.test(model, datamodule=dm)
