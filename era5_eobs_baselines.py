# Standard library
from argparse import ArgumentParser

import pytorch_lightning as pl

# Third party
import torch
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
from src.climate_learn.transforms import Mask, Denormalize
from src.climate_learn import load_downscaling_module
from src.climate_learn.data import IterDataModule

torch.set_float32_matmul_precision("medium")
parser = ArgumentParser()
parser.add_argument(
    "--era5_dir",
    type=str,
    # default="/app/data/processed/era5-eobs/era5_0.25_D"
    default="/app/data/processed/era5-eobs/e-obs/ensemble_mean/025_grid"
    )
parser.add_argument(
    "--eobs_dir",
    type=str,
    default="/app/data/processed/era5-eobs/e-obs/ensemble_mean/0125_grid_new"
    )


# Set up data
parser.add_argument(
    "--in_vars",
    type=list,
    default=[
        # "2m_temperature",
        # "maximum_temperature",
        # "minimum_temperature",
        # "rainfall"]
        "tg",
        "tx",
        "tn",
        "rr"]
    )

parser.add_argument(
    "--out_vars",
    type=list,
    default=[
        "tg",
        "tx",
        "tn",
        "rr"]
    )
args = parser.parse_args()


dm = IterDataModule(
    task="downscaling",
    inp_root_dir=args.era5_dir,
    out_root_dir=args.eobs_dir,
    in_vars=args.in_vars,
    out_vars=args.out_vars,
    subsample=1,
    batch_size=256,
    buffer_size=10000,
    num_workers=4,
)
dm.setup()

# Set up masking
mask = Mask(dm.out_mask)
denorm = Denormalize(dm)
denorm_mask = lambda x: denorm(mask(x))

# Set up baseline models
nearest = load_downscaling_module(
    data_module=dm,
    architecture="nearest-interpolation",
    train_target_transform=mask,
    val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
    test_target_transform=[denorm_mask, denorm_mask, denorm_mask]
)
bilinear = load_downscaling_module(
    data_module=dm,
    architecture="bilinear-interpolation",
    train_target_transform=mask,
    val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
    test_target_transform=[denorm_mask, denorm_mask, denorm_mask],
)
bicubic = load_downscaling_module(
    data_module=dm,
    architecture="bicubic-interpolation",
    train_target_transform=mask,
    val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
    test_target_transform=[denorm_mask, denorm_mask, denorm_mask]
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
    [nearest,
     bilinear,
     bicubic
     ],
    ["nearest-interpolation",
     "bilinear-interpolation",
     "bicubic-interpolation"
     ],
):
    print("Validating model:", model_name)
    trainer.validate(model, dataloaders=dm)

    print("Testing model:", model_name)
    trainer.test(model, dataloaders=dm)