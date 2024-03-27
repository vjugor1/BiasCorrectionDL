# Standard library
from argparse import ArgumentParser

# Third party
import torch
from src.climate_learn import LitModule, IterDataModule, load_downscaling_module
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

torch.set_float32_matmul_precision("medium")

parser = ArgumentParser()
parser.add_argument(
    "era5_low_res_dir",
    type=str,
    default="/app/data/ClimateLearn/processed/ERA5/5.625_24sh",
)
parser.add_argument(
    "era5_high_res_dir",
    type=str,
    default="/app/data/ClimateLearn/processed/ERA5/2.8125_24sh",
)
parser.add_argument(
    "architecture", type=str, choices=["resnet", "unet", "vit"], default="vit"
)
parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--gpu", type=int, default=[0, 1, 2, 3])
parser.add_argument("--checkpoint", default=None)
args = parser.parse_args()


# Set up data
in_vars = out_vars = [
    "2m_temperature",
]

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
    train_loss="mse",
    val_loss=["rmse", "pearson", "mean_bias", "mse"],
    test_loss=["rmse", "pearson", "mean_bias", "mse"],
    train_target_transform=None,
    val_target_transform=["denormalize", "denormalize", "denormalize", None],
    test_target_transform=["denormalize", "denormalize", "denormalize", None],
)

# Setup trainer
pl.seed_everything(0)
default_root_dir = (
    f"data/ClimateLearn/processed/downscaling-ERA5-ERA5/vit_downscaling_t2m"
)
logger = TensorBoardLogger(save_dir=f"{default_root_dir}/logs")
early_stopping = "val/mse:aggregate"
callbacks = [
    RichProgressBar(),
    RichModelSummary(max_depth=args.summary_depth),
    EarlyStopping(monitor=early_stopping, patience=args.patience),
    ModelCheckpoint(
        dirpath=f"{default_root_dir}/checkpoints",
        monitor=early_stopping,
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
    ),
]
trainer = pl.Trainer(
    enable_progress_bar=True,
    logger=logger,
    callbacks=callbacks,
    default_root_dir=default_root_dir,
    accelerator="gpu",
    devices=[0],
    max_epochs=args.max_epochs,
    strategy="auto",
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
