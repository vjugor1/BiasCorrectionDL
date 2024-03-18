import sys,os
sys.path.append(os.getcwd())
import warnings
warnings.filterwarnings("ignore")
import torch
import random
import logging
from datetime import datetime 
import pytorch_lightning as pl
from src.regression.models.pl_module import WindNetPL
from src.regression.datamodule import WindDataModule
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
import wandb
import time
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

def get_rundir_name() -> str:
    now = datetime.now()
    return str(f'out/{now:%Y-%m-%d}/{now:%H-%M-%S}')
    
@rank_zero_only
def log_config(cfg):
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

@rank_zero_only
def log_model_arch(model):
    logging.info(model)

def train_regression(cfg: DictConfig) -> None: 
    logging.info(f"Starting in {os.getcwd()}")
    start_time = time.process_time() 
    
    os.environ['WANDB_MODE'] = 'online'
    os.environ['WANDB_DIR'] = 'out/wandb'
    os.environ['WANDB_CONFIG_DIR'] = 'out/wandb'
    os.environ['WANDB_CACHE_DIR'] = 'out/wandb'
    torch.set_float32_matmul_precision('high')
    run_dir = get_rundir_name()  
    wandb_logger = WandbLogger(save_dir=os.path.join(os.getcwd(), run_dir),
                               project=cfg.project_name,
                               name=cfg.experiment_name,
                               log_model='all')
    dm = WindDataModule(cfg)
    model = WindNetPL(cfg, run_dir)
    logging.info(f"Asking for {cfg.train.gpu_num} GPUs")
    logging.info(f"Visible is {torch.cuda.device_count()} GPUs")
    logging.info(f"torch version {torch.__version__ }")

    if torch.__version__ == "2.0.1" or torch.__version__ == "2.0.0" or  torch.__version__ == "2.0.1+cu117":
        # model.net = torch.compile(model.net)
        logging.info("Model compiled") 
    else:
        logging.info("PyTorch version is smaller than 2.0, compilation is not supported")
        
    default_root_dir = run_dir
    checkpoint_loc = run_dir    
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_loc, save_top_k=2, monitor="val/loss")
    SWA = StochasticWeightAveraging(swa_lrs=0.004, swa_epoch_start=0.8, annealing_epochs=6)

    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=False)
    
    trainer = pl.Trainer(max_epochs=cfg.train.max_epoch,                         
                         default_root_dir=default_root_dir,
                         callbacks=[lr_monitor, checkpoint_callback, SWA],
                         #performance
                         accelerator="gpu",
                         precision="32",
                         benchmark=True,
                         #validation
                         check_val_every_n_epoch=1,
                         num_sanity_val_steps=0,
                         #distributed
                         devices=cfg.train.gpu_num,
                         num_nodes=cfg.train.num_nodes if cfg.train.distributed else 1,
                         strategy=cfg.train.strategy if cfg.train.distributed else 'auto',
                         #log
                         log_every_n_steps=cfg.train.log_every_n_steps,
                         logger=wandb_logger,
                         #misc
                         profiler='simple',
                         )   
    # log_config(cfg)
    logging.info(f"Time to start train {time.process_time() - start_time} seconds")
    trainer.fit(model, dm)
    

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="cmip6_GhostWindNet27.yaml")
def main(cfg: DictConfig):    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')
    train_regression(cfg)
    logging.info('Train finished!')


if __name__ == "__main__":    
    sys.argv.append('hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
    wandb.finish()
