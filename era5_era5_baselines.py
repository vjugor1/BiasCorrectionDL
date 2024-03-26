# Third party
import climate_learn as cl
import pytorch_lightning as pl

# Set up data
in_vars = out_vars = [
    "2m_temperature",

]
dm = cl.data.IterDataModule(
    "downscaling",
    inp_root_dir="/app/data/ClimateLearn/processed/ERA5/5.625_24sh",
    out_root_dir="/app/data/ClimateLearn/processed/ERA5/2.8125_24sh",
    in_vars=in_vars,
    out_vars=out_vars,
    subsample=1,
    batch_size=32,
    num_workers=4,
    )
dm.setup()

# Set up baseline models
nearest = cl.load_downscaling_module(data_module=dm, architecture="nearest-interpolation")
bilinear = cl.load_downscaling_module(data_module=dm, architecture="bilinear-interpolation")

# Evaluate baselines (no training needed)
trainer = pl.Trainer(accelerator="cpu")
trainer.test(nearest, dm)
trainer.test(bilinear, dm)