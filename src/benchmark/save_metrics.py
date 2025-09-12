# Standard library
import os
import sys
sys.path.append('/app')
import hydra
from hydra import compose, initialize
import glob
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import pickle

CONFIG_NAME = "cmip6-cmip6"
CONFIG_NAME_INFERENCE = "cmip6-cmip6_factor4"

if CONFIG_NAME == "cmip6-cmip6":
    from cmip6_cmip6_dl import main
elif CONFIG_NAME == "era5-era5":
    from era5_era5_dl import main
elif CONFIG_NAME == "eobs-eobs":
    from eobs_eobs_dl import main
elif CONFIG_NAME == "era5-eobs":
    from era5_eobs_dl import main

print(os.getcwd())
with initialize(version_base=None, config_path="../../configs/inference"):
                cfg = compose(config_name=CONFIG_NAME_INFERENCE)

with initialize(version_base=None, config_path="../../configs/train"):
                cfg_train = compose(config_name=CONFIG_NAME)
                
                
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

                        
def save_metrics():
    path = cfg.path
    models = cfg.models
    metrics = cfg.metrics
    
    # Create empty dataframe
    df = pd.DataFrame(columns = ["metric"] + models)
    df.set_index("metric", inplace=True)
    os.makedirs(os.path.join(cfg.path, "plots"), exist_ok=True)
    
    # Load metric values over all seeds and models
    for i, seed in enumerate(cfg.seeds):
        df_seed = df.copy()
        cfg_train["training"].seed = seed
        
        for model in models:
            # Test from checkpoint
            version=cfg[model][1].versions[i]
            ckpt_dir = glob.glob(os.path.join(path, f"{model}_multi_{cfg[model][0].upsampling}_{seed}/logs/version_{version}/checkpoints/"+"epoch_*.*"))
            cfg_train["model"].upsampling = cfg[model][0].upsampling
            cfg_train.model.architecture = model
            print(ckpt_dir)
            cfg_train.training.checkpoint = ckpt_dir[-1]
            print(cfg_train.training.checkpoint)
            main(cfg_train)
            
            # Load what that test did
            light_dirs = glob.glob(os.path.join(path, f"{model}_multi_{cfg[model][0].upsampling}_{seed}/logs/lightning_logs/")+"/version_*")
            light_dirs.sort(key=lambda x: int(x.split('_')[-1]))
            event_dir = light_dirs[-1]
            print("event_dir: ", event_dir)
            event_files = glob.glob(event_dir+"/events*.*")
            event_files.sort(key=lambda x: os.path.getmtime(x))
            metric_dict = parse_tensorboard(os.path.join(event_dir, event_files[-1]), metrics)
            
            for m in metrics: 
                row = metric_dict[m]
                row.name = m
                df_seed.at[m, model] = row.loc[0]["value"]
        
        # Save df with metrics of current seed
        df_seed.to_pickle(os.path.join(cfg.path, "plots", f"metrics_{seed}_gauss_06.pkl"))
    
    # # Average over all seeds
    for i, seed in enumerate(cfg.seeds):
        df_seed = pd.read_pickle(os.path.join(cfg.path, "plots", f"metrics_{seed}_gauss_06.pkl"))
        df = pd.concat([df, df_seed])

    # # Save averaged values
    df_avg = df.groupby(level=0).mean()
    df_avg.to_pickle(os.path.join(cfg.path, "plots", f"metrics_avg_gauss_06.pkl"))


if __name__ == "__main__":
    save_metrics()