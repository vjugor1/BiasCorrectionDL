# Third party
import torch
import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

torch.set_float32_matmul_precision("medium")
dm = cl.data.IterDataModule(
    "downscaling",
    inp_root_dir="/app/data/ClimateLearn/processed/ERA5/5.625_24sh",
    out_root_dir="/app/data/ClimateLearn/processed/ERA5/2.8125_24sh",
    in_vars=["2m_temperature"],
    out_vars=["2m_temperature"],
    subsample=1,
    batch_size=256,
    buffer_size=10000,
    num_workers=64,
)
dm.setup()

# Set up deep learning model
model = cl.load_downscaling_module(
    data_module=dm,
    architecture="vit",
    train_loss="mse",
    val_loss=["mse"],
    test_loss=["mse"],
    train_target_transform=None,
    val_target_transform=[None],
    test_target_transform=[None],
)

# Setup trainer
pl.seed_everything(0)
default_root_dir = (
    f"data/ClimateLearn/processed/downscaling-ERA5-ERA5/vit_downscaling_t2m_24sh"
)
logger = TensorBoardLogger(save_dir=f"{default_root_dir}/logs")
early_stopping = "val/mse:aggregate"
callbacks = [
    # RichProgressBar(),
    RichModelSummary(max_depth=1),
    EarlyStopping(monitor=early_stopping, patience=5),
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
    devices=-1,
    max_epochs=100,
    strategy="ddp",
    precision="16-mixed",
)

# Train and evaluate model from scratch
trainer.fit(model, datamodule=dm)
trainer.test(model, datamodule=dm, ckpt_path="best")
