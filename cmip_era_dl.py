# Standard library
from argparse import ArgumentParser

from src.climate_learn.utils import load_downscaling_module
from src.climate_learn import convert_nc2npz, IterDataModule,LitModule
from src.climate_learn.models.hub import VisionTransformer, Interpolation

# Third party
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    LearningRateMonitor,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

torch.set_float32_matmul_precision("medium")


parser = ArgumentParser()
parser.add_argument(
    "--default_root_dir",
    type=str,
    default="/app/data/experiments/downscaling-CMIP-ERA/unet_test",
)
parser.add_argument(
    "--inp_root_dir",
    type=str,
    default="/app/data/cmip6_3h_processed"
)
parser.add_argument(
    "--out_root_dir",
    type=str,
    default="/app/data/era5_3h_processed"
)
parser.add_argument(
    "--upsampling",
    type=str,
    choices=["bilinear", "bicubic", "unet_upsampling", "unet_upsampling_bilinear"],
    default="bicubic",
)
parser.add_argument(
    "--architecture", type=str, choices=["resnet", "unet", "vit"], default="unet"
)

parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=10)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--gpu", type=list, default=[0])
parser.add_argument("--checkpoint", default=None)
args = parser.parse_args()

# Set up data
in_vars = ["air_temperature", "u_component_of_wind", "v_component_of_wind", "surface_pressure"]
out_vars=["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "surface_pressure"]

dm = IterDataModule(
    task="downscaling",
    inp_root_dir=args.inp_root_dir,
    out_root_dir=args.out_root_dir,
    in_vars=in_vars,
    out_vars=out_vars,
    subsample=1,
    batch_size=1,
    buffer_size=10000,
    num_workers=4
)
dm.setup()


# Set up masking
# mask = Mask(dm.get_out_mask().to(device=f"cuda:{args.gpu}"))
# denorm = Denormalize(dm)
# denorm_mask = lambda x: denorm(mask(x))

# Default ViT preset is optimized for ERA5 to ERA5 downscaling, so we
# modify the architecture for ERA5 to PRISM
model = load_downscaling_module(
    data_module=dm,
    architecture=args.architecture,
    # upsampling=args.upsampling,
    # model_kwargs={"neck_chans": 256},
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
    # strategy="ddp",
    precision="bf16-mixed",
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
        test_target_transforms=model.test_target_transforms,
    )
    trainer.test(model, datamodule=dm)