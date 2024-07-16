# Standard library
from argparse import ArgumentParser

# Third party
import torch
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
import pytorch_lightning as pl

from src.climate_learn import load_downscaling_module
from src.climate_learn.data import IterDataModule

torch.set_float32_matmul_precision("medium")

parser = ArgumentParser()
parser.add_argument(
    "--cmip6_low_res_dir",
    type=str,
    default="/app/data/processed/cmip6-cmip6/LR",
)
parser.add_argument(
    "--cmip6_high_res_dir",
    type=str,
    default="/app/data/processed/cmip6-cmip6/HR",
)
parser.add_argument(
    "--in_vars",
    type=list,
    default=[
        "air_temperature", "u_component_of_wind", "v_component_of_wind", "precipitation"
    ],  
)
parser.add_argument(
    "--out_vars",
    type=list,
    default=[
        "air_temperature", "u_component_of_wind", "v_component_of_wind", "precipitation"
    ],
)
args = parser.parse_args()

# Set up data
in_vars = args.in_vars
out_vars = args.out_vars

# Set up data
dm = IterDataModule(
    task="downscaling",
    inp_root_dir=args.cmip6_low_res_dir,
    out_root_dir=args.cmip6_high_res_dir,
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
bicubic = load_downscaling_module(data_module=dm, architecture="bicubic-interpolation")

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
    [nearest, bilinear, bicubic],
    ["nearest-interpolation", "bilinear-interpolation", "bicubic-interpolation"],
):
    print("Validating model:", model_name)
    trainer.validate(model, dataloaders=dm)

    print("Testing model:", model_name)
    trainer.test(model, dataloaders=dm)
