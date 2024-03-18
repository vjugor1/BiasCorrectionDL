import sys,os
sys.path.append(os.getcwd())
import warnings
import torch
import random
import logging
import pytorch_lightning as pl
from src.regression.models.pl_module import WindNetPL
from datamodule import WindDataModule
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
import wandb
import time
from pytorch_lightning.callbacks import LearningRateMonitor, OnExceptionCheckpoint
print(os.getcwd())
warnings.filterwarnings("ignore")

os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_DIR'] = 'out/wandb'
os.environ['WANDB_CONFIG_DIR'] = 'out/wandb'
os.environ['WANDB_CACHE_DIR'] = 'out/wandb'
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')
torch.set_float32_matmul_precision('high')

def get_rundir_name() -> str:
    now = datetime.now()
    return str(f'out/{now:%Y-%m-%d}/{now:%H-%M-%S}')

def test(cfg: DictConfig) -> None:        
    start_time = time.process_time()  
    run_dir = get_rundir_name()  

    wandb_logger = WandbLogger(save_dir=os.path.join(os.getcwd(), run_dir),
                               project=cfg.project_name,
                               name=cfg.experiment_name + '_test')
    dm = WindDataModule(cfg)
    model = WindNetPL(cfg, run_dir)
    # model.load_from_checkpoint(os.path.join(os.getcwd(), cfg.eval.path_to_checkpoint), cfg=cfg)

    wandb_logger.watch(model, log='all', log_freq=100)       
    trainer = pl.Trainer(max_epochs=cfg.train.max_epoch,
                         accelerator="gpu" if ((cfg.model_name != "BaselineQT") and (cfg.model_name != "BaselineQW")) else "cpu",
                         precision="32",
                         benchmark=True,
                         devices=cfg.eval.gpu_num,
                         default_root_dir=run_dir,
                         strategy=cfg.eval.strategy if cfg.eval.distributed_test else 'auto',
                         logger=wandb_logger,
                         #limit_test_batches=200
                        )
    
    logging.info(f"Time to start test {time.process_time() - start_time} seconds")
    if (cfg.model_name == "BaselineQT") or (cfg.model_name == "BaselineQW"):
        trainer.test(model, dm)
    else:
        trainer.test(model, dm, ckpt_path=os.path.join(os.getcwd(), cfg.eval.path_to_checkpoint))
 

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="cmip6_GhostWindNet27.yaml")
def main(cfg: DictConfig):    
    cfg.eval.time_start = cfg.start_date  
    cfg.eval.time_end = cfg.end_date
    test(cfg)
    logging.info('Test finished!')


if __name__ == "__main__":    
    sys.argv.append('hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
    wandb.finish()