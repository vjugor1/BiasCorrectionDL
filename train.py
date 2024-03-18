import sys,os
sys.path.append(os.getcwd())
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from src.regression.train import train_regression, get_rundir_name

    

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="cmip6_GhostWindNet27")
def main(cfg: DictConfig):    
    train_regression(cfg)
    logging.info('Train finished!')


if __name__ == "__main__":
    experiment_name = 'latest'
    rundir_name = get_rundir_name()
    sys.argv.append(f'hydra.run.dir={rundir_name}')
    main()
