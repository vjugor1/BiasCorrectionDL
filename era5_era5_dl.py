# Standard library
from argparse import ArgumentParser

# Third party
import torch
from src.climate_learn import LitModule, IterDataModule, load_downscaling_module
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

torch.set_float32_matmul_precision("medium")

parser = ArgumentParser()
parser.add_argument(
    "--default_root_dir",
    type=str,
    default="/app/data/ClimateLearn/experiments/downscaling-ERA5-ERA5/vit_downscaling_t2_t2_single_gpu_bilinear_sched",
)
parser.add_argument(
    "--era5_low_res_dir",
    type=str,
    default="/app/data/ClimateLearn/processed/ERA5/5.625",
)
parser.add_argument(
    "--era5_high_res_dir",
    type=str,
    default="/app/data/ClimateLearn/processed/ERA5/2.8125",
)
parser.add_argument(
    "--architecture", type=str, choices=["resnet", "unet", "vit"], default="vit"
)
parser.add_argument(
    "--upsampling",
    type=str,
    choices=["bilinear", "bicubic", "unet_upsampling", "unet_upsampling_bilinear"],
    default="bilinear",
)
parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--gpu", type=list, default=[0,1,2,3])
parser.add_argument("--checkpoint", default=None)
args = parser.parse_args()


# Set up data
in_vars = ["2m_temperature", "temperature_850", "geopotential_500"]
out_vars = ["2m_temperature", "temperature_850", "geopotential_500"]

dm = IterDataModule(
    task="downscaling",
    inp_root_dir=args.era5_low_res_dir,
    out_root_dir=args.era5_high_res_dir,
    in_vars=in_vars,
    out_vars=out_vars,
    subsample=1,
    batch_size=256,
    buffer_size=10000,
    num_workers=4,
)
dm.setup()

# Set up deep learning model
model = load_downscaling_module(
    data_module=dm,
    architecture=args.architecture,
    upsampling=args.upsampling,
    optim_kwargs={"lr": 5e-4},
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
pl.seed_everything(0)
logger = TensorBoardLogger(save_dir=f"{args.default_root_dir}/logs")
early_stopping = "val/mse:aggregate"
callbacks = [
    RichProgressBar(),
    RichModelSummary(max_depth=args.summary_depth),
    EarlyStopping(monitor=early_stopping, patience=args.patience),
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
    strategy="ddp",
    precision="16-mixed",
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
