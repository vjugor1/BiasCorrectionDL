# Standard library
from hydra import compose, initialize

# Third party
import torch
import pickle
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
import pytorch_lightning as pl

from src.climate_learn.transforms import Mask, Denormalize
from src.climate_learn import load_downscaling_module
from src.climate_learn.data import IterDataModule

torch.set_float32_matmul_precision("medium")

def load_model_baseline():
    with initialize(version_base=None, config_path="configs/train"):
        cfg_train = compose(config_name="era5-eobs")
    
    # Set up data
    in_vars = ["2m_temperature",
            "maximum_temperature",
            "minimum_temperature",
            "rainfall"]
    out_vars = cfg_train.data.out_variables


    dm = IterDataModule(
        task="downscaling",
        inp_root_dir=cfg_train.data.era5_low_res_dir,
        out_root_dir=cfg_train.data.eobs_high_res_dir,
        in_vars=in_vars,
        out_vars=out_vars,
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
    # nearest = load_downscaling_module(
    #     data_module=dm,
    #     architecture="nearest-interpolation",
    #     train_target_transform=mask,
    #     val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask, denorm_mask, denorm_mask, denorm_mask],
    #     test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask, denorm_mask, denorm_mask],
    # )
    bilinear = load_downscaling_module(
        data_module=dm,
        architecture="bilinear-interpolation",
        train_target_transform=mask,
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask, denorm_mask, denorm_mask, denorm_mask],
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask, denorm_mask, denorm_mask],
    )
    bicubic = load_downscaling_module(
        data_module=dm,
        architecture="bicubic-interpolation",
        train_target_transform=mask,
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask, denorm_mask, denorm_mask, denorm_mask],
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask, denorm_mask, denorm_mask, denorm_mask]
    )

    callbacks = [
        RichProgressBar(),
        RichModelSummary(max_depth=1),
    ]
    # Evaluate baselines (no training needed)
    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=callbacks,
    )
     
    return dm, trainer, (bilinear, bicubic)

if __name__ == "__main__":
    dm, trainer, (bilinear, bicubic) = load_model_baseline()
    
    # Perform validation and testing for each model
    for model, model_name in zip(
        [bilinear, bicubic],
        ["bilinear-interpolation", "bicubic-interpolation"],
    ):
        print("Validating model:", model_name)
        trainer.validate(model, dataloaders=dm)

        print("Testing model:", model_name)
        trainer.test(model, dataloaders=dm)