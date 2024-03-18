import sys,os
sys.path.append(os.getcwd())

from datetime import datetime 
import pytorch_lightning as pl
from src.regression.models.pl_module import WindNetPL
from src.regression.datamodule import WindDataModule
import hydra
from torchsummary import summary


def get_rundir_name() -> str:
    now = datetime.now()
    return str(f'out/{now:%Y-%m-%d}/{now:%H-%M-%S}')
    

def train_regression(cfg) -> None: 
    run_dir = get_rundir_name()  

    model = WindNetPL(cfg, run_dir)
    clim = [3, 27, 95, 95]
    elev = [1, 2769, 2769]
    summary(model, clim)

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="cmip5_WindNet27x47.yaml")
def main(cfg):    
    train_regression(cfg)


if __name__ == "__main__":    
    sys.argv.append('hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()

