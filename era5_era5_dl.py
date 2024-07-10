# Standard library
import os
from omegaconf import DictConfig, OmegaConf
import hydra

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

torch.set_float32_matmul_precision("high")

@hydra.main(config_path="/app/configs/train", config_name="era5-era5")
def main(cfg: DictConfig):
    # Construct dynamic experiment name
    experiment_name = construct_experiment_name(cfg)
    default_root_dir  = os.path.join(cfg.base_dir, experiment_name)

    # Set the seed for reproducibility
    pl.seed_everything(cfg.training.seed)

    dm = setup_data_module(cfg)
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
    if config.model.architecture == "diffusion":
        for var in out_vars:
            in_vars.remove(var)
        in_vars = out_vars + in_vars
        assert out_vars == in_vars[:len(out_vars)], "Out variables' names (`out_vars`) must be placed in the beginning of `in_vars`"
        
    dm = IterDataModule(
        task="downscaling",
        inp_root_dir=config.data.era5_low_res_dir,
        out_root_dir=config.data.era5_high_res_dir,
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
        optim_kwargs={"lr": config.training.learning_rate},
        sched="linear-warmup-cosine-annealing",
        sched_kwargs={
            "warmup_epochs": config.training.warmup_epochs,
            "max_epochs": config.training.max_epochs,
        },
        train_loss="perceptual",
        val_loss=["rmse", "pearson", "mean_bias", "mse", "perceptual"],
        test_loss=["rmse", "pearson", "mean_bias", "perceptual"],
        train_target_transform=None,
        val_target_transform=["denormalize", "denormalize", "denormalize", None, "denormalize"],
        test_target_transform=["denormalize", "denormalize", "denormalize", "denormalize"],
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