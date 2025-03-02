# Standard library
import os
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np

import pytorch_lightning as pl
# Third party
import torch
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, RichModelSummary,
                                         RichProgressBar)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from src.climate_learn import (IterDataModule, LitModule,
                               load_downscaling_module)

from src.climate_learn.data.processing.era5_constants import (
    DEFAULT_PRESSURE_LEVELS, PRESSURE_LEVEL_VARS)
from src.climate_learn.transforms import Mask, Denormalize

torch.set_float32_matmul_precision("medium")

@hydra.main(config_path="/app/configs/train", config_name="era5-eobs")
def main(cfg: DictConfig):
    # Construct dynamic experiment name
    experiment_name = construct_experiment_name(cfg)
    default_root_dir  = os.path.join(cfg.base_dir, experiment_name)

    # Set the seed for reproducibility
    pl.seed_everything(cfg.training.seed)

    dm = setup_data_module(cfg) 
    
    # Set up masking
    dm.mask = Mask(dm.out_mask)
    denorm = Denormalize(dm)
    dm.denorm_mask = lambda x: denorm(dm.mask(x))
    
    model = setup_model(dm, cfg)
    trainer = setup_trainer(cfg, default_root_dir)

    # Train and evaluate model from scratch
    if cfg.training.checkpoint is None:
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm, ckpt_path="best")
    # Evaluate saved model checkpoint
    else:
        model = LitModule.load_from_checkpoint(
            cfg.training.checkpoint,
            net=model.net,
            optimizer=model.optimizer,
            lr_scheduler=None,
            train_loss=None,
            val_loss=None,
            test_loss=model.test_loss,
            test_target_transforms=model.test_target_transforms,
        )
        trainer.test(model, datamodule=dm)

def construct_experiment_name(config):
    architecture = config.model.architecture
    upsampling = config.model.upsampling
    out_variables = list(config.data.out_variables)
    seed = config.training.seed
    mode = "single"
    if len(out_variables) > 1:
        mode = "multi"
    experiment_name = f"{architecture}_{mode}_{upsampling}_{seed}"
    return experiment_name

def setup_data_module(config):
    variables = config.data.in_variables
    in_vars = []
    for var in variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                in_vars.append(var + "_" + str(level))
        else:
            in_vars.append(var)
    out_vars = config.data.out_variables
    vars_corr = np.load(os.path.join(
        os.path.dirname(config.data.era5_low_res_dir), "vars.npy"), allow_pickle=True).item()
    
    if config.model.architecture in ["diffusion", "dcgan"]:
        out_vars_new=[]
        for var in out_vars:
            in_vars.remove(vars_corr[var])
            out_vars_new.append(vars_corr[var])
        in_vars = out_vars_new + in_vars
        assert out_vars_new == in_vars[:len(out_vars)], "Out variables' names (`out_vars`) must be placed in the beginning of `in_vars`"
        
    dm = IterDataModule(
        task="downscaling",
        inp_root_dir=config.data.era5_low_res_dir,
        out_root_dir=config.data.eobs_high_res_dir,
        in_vars=in_vars,
        out_vars=out_vars,
        subsample=config.data.subsample,
        batch_size=config.data.batch_size,
        buffer_size=config.data.buffer_size,
        num_workers=config.data.num_workers,
    )
    dm.setup()
    return dm

def setup_model(dm, config):
    model = load_downscaling_module(
        data_module=dm,
        architecture=config.model.architecture,
        upsampling=config.model.upsampling,
        optim_kwargs={"lr": config.training.learning_rate,
                      "weight_decay": config.training.weight_decay,
                      "betas": tuple(config.training.betas),
                      },
        sched="linear-warmup-cosine-annealing",
        sched_kwargs={
            "warmup_epochs": config.training.warmup_epochs,
            "max_epochs": config.training.max_epochs,
        },
        train_loss=tuple(config.training.train_loss) if len(config.training.train_loss) > 1 else str(config.training.train_loss[0]),
        train_target_transform=dm.mask,
        val_target_transform=[dm.denorm_mask, dm.denorm_mask, dm.denorm_mask, dm.mask, dm.denorm_mask, dm.denorm_mask, dm.denorm_mask],
        test_target_transform=[dm.denorm_mask, dm.denorm_mask, dm.denorm_mask, dm.denorm_mask, dm.denorm_mask, dm.denorm_mask]
    )
    return model

def setup_trainer(config, default_root_dir):
    logger = TensorBoardLogger(save_dir=f"{default_root_dir}/logs")
    early_stopping = config.training.early_stopping
    callbacks = [
        RichProgressBar(),
        RichModelSummary(max_depth=config.training.summary_depth),
        EarlyStopping(
            monitor=early_stopping,
            min_delta=config.training.min_delta,
            patience=config.training.patience,
            verbose=True,
            mode="min",
        ),
        ModelCheckpoint(
            dirpath=os.path.join(f"{default_root_dir}/logs", f"version_{logger.version}", "checkpoints"),
            monitor=early_stopping,
            filename="epoch_{epoch:03d}",
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = pl.Trainer(
        # accumulate_grad_batches=2,
        enable_progress_bar=True,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=default_root_dir,
        accelerator="gpu",
        devices=config.training.gpus,
        max_epochs=config.training.max_epochs,
        precision=config.training.precision,
    )
    return trainer

if __name__ == "__main__":
    main()