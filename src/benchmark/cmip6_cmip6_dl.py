# Standard library
import os, sys
from omegaconf import DictConfig, OmegaConf
import hydra
import pickle

# Third party
import torch
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, RichModelSummary,
                                         RichProgressBar)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import pytorch_lightning as pl

from src.climate_learn import (IterDataModule, LitModule,
                               DiffusionLitModule, DeepSDLitModule,
                               YnetLitModule, GANLitModule, ESRGANLitModule,
                               load_downscaling_module)
from src.climate_learn.utils.gis import prepare_ynet_climatology, prepare_deepsd_elevation, prepare_dcgan_elevation
from src.climate_learn.data.processing.era5_constants import (
    DEFAULT_PRESSURE_LEVELS, PRESSURE_LEVEL_VARS)
from src.climate_learn.transforms import add_iid_gaussian

torch.set_float32_matmul_precision("medium")

@hydra.main(config_path="../../configs/train", config_name="cmip6-cmip6")
def main(cfg: DictConfig):
    # Construct dynamic experiment name
    experiment_name = construct_experiment_name(cfg)
    default_root_dir  = os.path.join(cfg.base_dir, experiment_name)

    os.makedirs(f'{default_root_dir}/logs/', exist_ok=True)

    with open(f"{default_root_dir}/logs/config_of_experiment.dictconfig.pickle", "wb") as fd:
        pickle.dump(cfg, fd)
    
    # Set the seed for reproducibility
    pl.seed_everything(cfg.training.seed)

    dm = setup_data_module(cfg)
    model = setup_model(dm, cfg)
    trainer = setup_trainer(cfg, default_root_dir)
    
    path_to_elevation = "/app/data/elevation.nc"
    out_vars = cfg.data.out_variables
    
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
        if cfg.model.architecture == 'diffusion':
            model_module = DiffusionLitModule.load_from_checkpoint(
                cfg.training.checkpoint,
                net=model.net,
                optimizer=model.optimizer,
                lr_scheduler=None,
                train_loss=None,
                val_loss=None,
                test_loss=model.test_loss,
                test_target_transforms=model.test_target_transforms,
                test_in_transform=add_iid_gaussian # Это жесткий хардкод, надо потом исправить
            )
        elif cfg.model.architecture == "ynet":
            normalized_clim = prepare_ynet_climatology(dm, path_to_elevation, out_vars)
            
            model_module = YnetLitModule.load_from_checkpoint(
                cfg.training.checkpoint,
                net=model.net,
                optimizer=model.optimizer,
                lr_scheduler=None,
                train_loss=None,
                val_loss=None,
                test_loss=model.test_loss,
                test_target_transforms=model.test_target_transforms,
                x_aux = normalized_clim,
                test_in_transform=add_iid_gaussian # Это жесткий хардкод, надо потом исправить
            )
        elif cfg.model.architecture == "deepsd":
            elevation_list = prepare_deepsd_elevation(dm, path_to_elevation)
            
            model_module = DeepSDLitModule.load_from_checkpoint(
                cfg.training.checkpoint,
                net=model.net,
                optimizer=model.optimizer,
                lr_scheduler=None,
                train_loss=None,
                val_loss=None,
                test_loss=model.test_loss,
                test_target_transforms=model.test_target_transforms,
                elevation = elevation_list,
            )
            
        elif cfg.model.architecture == "dcgan":
            elevation = prepare_dcgan_elevation(dm, path_to_elevation)
            model_module = GANLitModule.load_from_checkpoint(
                cfg.training.checkpoint,
                net=model.net,
                optimizer=model.optimizer,
                lr_scheduler=None,
                train_loss=None,
                val_loss=None,
                test_loss=model.test_loss,
                test_target_transforms=model.test_target_transforms,
                elevation = elevation

            )
        elif cfg.model.architecture == "esrgan":
            model_module = ESRGANLitModule.load_from_checkpoint(
                cfg.training.checkpoint,
                net=model.net,
                optimizer=model.optimizer,
                lr_scheduler=None,
                train_loss=None,
                val_loss=None,
                test_loss=model.test_loss,
                test_target_transforms=model.test_target_transforms,
                test_in_transform=add_iid_gaussian # Это жесткий хардкод, надо потом исправить
            )
        else:
            model_module = LitModule.load_from_checkpoint(
                cfg.training.checkpoint,
                net=model.net,
                optimizer=model.optimizer,
                lr_scheduler=None,
                train_loss=None,
                val_loss=None,
                test_loss=model.test_loss,
                test_target_transforms=model.test_target_transforms,
                test_in_transform=add_iid_gaussian # Это жесткий хардкод, надо потом исправить
            )

        trainer.test(model_module, datamodule=dm)

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
        test_in_transform=add_iid_gaussian if config.training.add_input_noise==True else None 
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
        # accumulate_grad_batches=4,
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