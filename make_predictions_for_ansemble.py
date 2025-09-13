import glob
import os
import torch
import tqdm
import pickle

from hydra import compose, initialize
from cmip6_cmip6_dl_base_model import main

CONFIG_NAME_TRAIN = "cmip6-cmip6"
CONFIG_NAME_INFERENCE = "cmip6-cmip6_factor4"

with initialize(version_base=None, config_path="configs/inference"):
    cfg_inference = compose(config_name=CONFIG_NAME_INFERENCE)

with initialize(version_base=None, config_path="configs/train"):
    cfg_train = compose(config_name=CONFIG_NAME_TRAIN)

train_ens_data = {}
val_ens_data = {}
test_ens_data = {}

i = 0
seed = 42
model = 'diffusion'
device = 'cuda:3'

version=cfg_inference[model][1].versions[i]
ckpt_dir = glob.glob(os.path.join(cfg_inference.path, f"{model}_multi_{cfg_inference[model][0].upsampling}_{seed}/logs/version_{version}/checkpoints/"+"epoch_*.*"))
cfg_train["model"].upsampling = cfg_inference[model][0].upsampling
cfg_train.model.architecture = model
cfg_train.training.checkpoint = ckpt_dir[-1]

model_module, dm = main(cfg_train);
model_module.eval().to(device)
dm.setup()

y_pred_train = []
with torch.no_grad():
    for batch in tqdm.tqdm(dm.predict_dataloader()):
        x, _, _, _ = batch
        y_pred_train.append(model_module(x.to(device)))

y_pred_val = []
with torch.no_grad():
    for batch in dm.val_dataloader():
        x, _, _, _ = batch
        y_pred_val.append(model_module(x.to(device)))

y_pred_test = []
with torch.no_grad():
    for batch in dm.test_dataloader():
        x, _, _, _ = batch
        y_pred_test.append(model_module(x.to(device)))

train_ens_data[model + '_' + str(seed)] = torch.cat(y_pred_train).to('cpu')
val_ens_data[model + '_' + str(seed)] = torch.cat(y_pred_val).to('cpu')
test_ens_data[model + '_' + str(seed)] = torch.cat(y_pred_test).to('cpu')

print('saving train')
with open(f'train_ens_data_factor_4_{model}_{seed}.pkl', 'wb') as f:
    pickle.dump(train_ens_data, f)
print('done')

print('saving val')
with open(f'val_ens_data_factor_4_{model}_{seed}.pkl', 'wb') as f:
    pickle.dump(val_ens_data, f)
print('done')

print('saving test')
with open(f'test_ens_data_factor_4_{model}_{seed}.pkl', 'wb') as f:
    pickle.dump(test_ens_data, f)
print('done')
