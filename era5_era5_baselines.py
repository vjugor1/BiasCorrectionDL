# Standard library
from argparse import ArgumentParser

# Third party
import torch
from src.climate_learn.data import IterDataModule
from src.climate_learn import load_downscaling_module
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    RichModelSummary,
    RichProgressBar,
)

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
args = parser.parse_args()

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

# Set up baseline models
nearest = load_downscaling_module(data_module=dm, architecture="nearest-interpolation")
bilinear = load_downscaling_module(
    data_module=dm, architecture="bilinear-interpolation"
)

callbacks = [
    RichProgressBar(),
    RichModelSummary(max_depth=1),
]
# Evaluate baselines (no training needed)
trainer = pl.Trainer(
    accelerator="cpu",
    callbacks=callbacks,
)


# Perform validation and testing for each model
for model, model_name in zip(
    [nearest, bilinear], ["nearest-interpolation", "bilinear-interpolation"]
):
    print("Validating model:", model_name)
    trainer.validate(model, dataloaders=dm)

    print("Testing model:", model_name)
    trainer.test(model, dataloaders=dm)
