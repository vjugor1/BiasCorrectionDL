# Standard library
import os
import sys
sys.path.append('/app')

from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import glob 
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd


def parse_tensorboard(path: str,
                      scalars: list):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}

@hydra.main(config_path="/app/configs/inference", config_name="era5_era5", version_base = None)
def save_metrics(cfg: DictConfig):
    path = cfg.path    
    models = cfg.models
    metrics=cfg.metrics
    df = pd.DataFrame(columns = ["metric"] + models)
    df.set_index("metric", inplace=True)
    
    
    for i, seed in enumerate(cfg.seeds):
        df_seed = df.copy()
        for model in models:
            version=cfg[model][1].versions[i]
            event_dir = os.path.join(path, f"{model}_multi_{cfg[model][0].upsampling}_{seed}/logs/lightning_logs/version_{version}")
            event_files = glob.glob(event_dir+"/events*.*")
            latest_event_file = max(event_files, key=os.path.getctime)
            metric_dict = parse_tensorboard(os.path.join(event_dir, latest_event_file), metrics)
            
            for m in metrics:
                row = metric_dict[m]
                row.name = m
                df_seed.at[m, model] = row.loc[0]["value"]
        
        # Save dataframe
        df.to_pickle(os.path.join(cfg.path, "plots", f"metrics_{seed}.pkl"))
    
    # Average over all seeds
    for i, seed in enumerate(cfg.seeds):
        df_seed = pd.read_pickle(os.path.join(cfg.path, f"metrics_{seed}.pkl"))
        df = pd.concat([df, df_seed])

    df_avg = df.groupby(level=0).mean()
    df_avg.to_pickle(os.path.join(cfg.path, "plots", f"metrics_avg.pkl"))

    
if __name__ == "__main__":
    save_metrics()