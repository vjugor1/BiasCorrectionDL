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

torch.set_float32_matmul_precision("medium")

@hydra.main(config_path="/app/configs/train", config_name="cmip6-cmip6")
def main(cfg: DictConfig):
    # Construct dynamic experiment name
    experiment_name = construct_experiment_name(cfg)
    default_root_dir  = os.path.join(cfg.base_dir, experiment_name)

    # Set the seed for reproducibility
    pl.seed_everything(cfg.training.seed)

    dm = setup_data_module(cfg)
    model = setup_model(dm, cfg)
    trainer = setup_trainer(cfg, default_root_dir)
    try:
        model.train_loss.vgg = model.train_loss.vgg.to(cfg.training.gpus[0])
    except AttributeError:
        pass
    try:
        model.train_loss.weights_x = model.train_loss.weights_x.to(cfg.training.gpus[0])
        model.train_loss.weights_y = model.train_loss.weights_y.to(cfg.training.gpus[0])
        if '16' in cfg.training.precision:
            model.train_loss.weights_x = model.train_loss.weights_x.astype(torch.half)
            model.train_loss.weights_y = model.train_loss.weights_y.astype(torch.half)
        else:
            model.train_loss.weights_x = model.train_loss.weights_x.astype(torch.float)
            model.train_loss.weights_y = model.train_loss.weights_y.astype(torch.float)
    except AttributeError:
        pass
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
    in_vars = config.data.in_variables
    out_vars = config.data.out_variables
    
    dm = IterDataModule(
        task="downscaling",
        inp_root_dir=config.data.low_res_dir,
        out_root_dir=config.data.high_res_dir,
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
        train_loss_kwargs=config.training.perceptual_hp,
        val_loss=["rmse", "pearson", "mean_bias", "mse"],
        test_loss=["rmse", "pearson", "mean_bias"],
        train_target_transform=None,
        val_target_transform=["denormalize", "denormalize", "denormalize", None],
        test_target_transform=["denormalize", "denormalize", "denormalize"],
    )
    return model

def setup_trainer(config, default_root_dir):
    logger = TensorBoardLogger(save_dir=f"{default_root_dir }/logs")
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
        default_root_dir=default_root_dir ,
        accelerator="gpu",
        devices=config.training.gpus,
        max_epochs=config.training.max_epochs,
        precision=config.training.precision,
    )
    return trainer

if __name__ == "__main__":
    main()