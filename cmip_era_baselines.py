# Standard library
from argparse import ArgumentParser
import pytorch_lightning as pl

from src.climate_learn.utils import load_downscaling_module
from src.climate_learn import convert_nc2npz, IterDataModule

parser = ArgumentParser()
parser.add_argument("inp_root_dir")
parser.add_argument("out_root_dir")
parser.add_argument("in_vars")
parser.add_argument("out_vars")
args = parser.parse_args()

# Set up data
dm = IterDataModule(
    task="downscaling"
    inp_root_dir = args.inp_root_dir,
    out_root_dir = args.out_root_dir,
    in_vars = args.in_vars,
    out_vars = args.out_vars,
    batch_size=256,
    num_workers=4,
)
dm.setup()

# Set up baseline models
nearest = load_downscaling_module(
    data_module=dm,
    architecture="nearest-interpolation",
)
bilinear = load_downscaling_module(
    data_module=dm,
    architecture="bilinear-interpolation",
)

# Evaluate baselines (no training needed)
trainer = pl.Trainer()
trainer.test(nearest, dm)
trainer.test(bilinear, dm)