import glob
import os
import torch
import tqdm
import pickle

import sys
sys.path.append('/app')

from hydra import compose, initialize
from cmip6_cmip6_dl_base_model import main

FACTOR = 2
CONFIG_NAME_TRAIN = "cmip6-cmip6"
CONFIG_NAME_INFERENCE = f"cmip6-cmip6_factor{FACTOR}_ens"

with initialize(version_base=None, config_path="../../configs/inference"):
    cfg_inference = compose(config_name=CONFIG_NAME_INFERENCE)

with initialize(version_base=None, config_path="../../configs/train"):
    cfg_train = compose(config_name=CONFIG_NAME_TRAIN)

train_ens_data = {}
val_ens_data = {}
test_ens_data = {}
i=1
models = cfg_inference.models_base
seed = cfg_train.training.seed
device = 'cuda:2'

for model in models:
    version=cfg_inference[model][1].versions[i]
    ckpt_dir = glob.glob(os.path.join(cfg_inference.path, f"{model}_multi_{cfg_inference[model][0].upsampling}_{seed}/logs/version_{version}/checkpoints/"+"epoch_*.*"))
    cfg_train["model"].upsampling = cfg_inference[model][0].upsampling
    cfg_train.model.architecture = model
    print(f"{model}_multi_{cfg_inference[model][0].upsampling}_{seed}/logs/version_{version}/checkpoints/"+"epoch_*.*")
    cfg_train.training.checkpoint = ckpt_dir[-1]

    model_module, dm = main(cfg_train)
    model_module.eval().to(device)
    dm.setup()

    for stage, dataloader in zip(['train', 'test', 'val'], 
                                [dm.train_dataloader(), dm.test_dataloader(), dm.val_dataloader()]
                                ):
        y_pred = []
        with torch.inference_mode():
            for batch in tqdm.tqdm(dataloader):
                x, _, _, _ = batch
                y_pred.append(model_module(x.to(device, non_blocking=True)))

        y_pred = torch.cat(y_pred, dim=0) #.to('cpu')

        print(f'saving {stage}')
        torch.save(y_pred, os.path.join(
                    os.path.dirname(cfg_train.data.high_res_dir),
                    f"preds_factor{FACTOR}",
                    f'{stage}_ens_data_factor_{FACTOR}_{model}_{seed}.pt'))
        print('done')