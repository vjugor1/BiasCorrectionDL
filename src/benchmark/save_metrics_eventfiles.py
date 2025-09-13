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


@hydra.main(config_path="/app/configs/inference", config_name="cmip6-cmip6_ens", version_base = None)
def save_metrics(cfg: DictConfig):
    path = cfg.path    
    models = cfg.models
    metrics=cfg.metrics
    
    # Create empty dataframe
    df = pd.DataFrame(columns = ["metric"] + models)
    df.set_index("metric", inplace=True)
    os.makedirs(os.path.join(cfg.path, "plots"), exist_ok=True)
    
    # Load metric values over all seeds and models
    for i, seed in enumerate(cfg.seeds):
        df_seed = df.copy()
        for model in models:
            version=cfg[model][1].versions[i]
            event_dir = os.path.join(path, f"{model}_multi_{cfg[model][0].upsampling}_{seed}/logs/lightning_logs/version_{version}")
            event_files = glob.glob(event_dir+"/events*.*")
            event_files.sort(key=lambda x: os.path.getmtime(x))
            print(event_dir)
            metric_dict = parse_tensorboard(os.path.join(event_dir, event_files[-1]), metrics)
            
            for m in metrics: 
                row = metric_dict[m]
                row.name = m
                df_seed.at[m, model] = row.loc[0]["value"]
        
        # Save df with metrics of current seed
        df_seed.to_pickle(os.path.join(cfg.path, "plots", f"metrics_{seed}_temp.pkl"))
    
    # Average over all seeds
    for i, seed in enumerate(cfg.seeds):
        df_seed = pd.read_pickle(os.path.join(cfg.path, "plots", f"metrics_{seed}_temp.pkl"))
        df = pd.concat([df, df_seed])

    # Save averaged values
    df_avg = df.groupby(level=0).mean()
    df_avg.to_pickle(os.path.join(cfg.path, "plots", f"metrics_avg_temp.pkl"))

    
if __name__ == "__main__":
    save_metrics()