import sys,os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import fiona
from fiona.drvsupport import supported_drivers
from src.regression.eval import eval
from src.regression.risk_estimation import risk_estimation
supported_drivers['LIBKML'] = 'rw'
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="cmip6_GhostWindNet27.yaml")
def main(cfg: DictConfig): 
    eval(cfg)
    

if __name__ == "__main__":      
    sys.argv.append('hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
