import glob
import os
import torch
import tqdm
import pickle

from hydra import compose, initialize
from cmip6_cmip6_dl_base_model import main

CONFIG_NAME = "cmip6-cmip6"

with initialize(version_base=None, config_path="configs/inference"):
    cfg_inference = compose(config_name=CONFIG_NAME)

with initialize(version_base=None, config_path="configs/train"):
    cfg_train = compose(config_name=CONFIG_NAME)

train_ens_data = {}
val_ens_data = {}
test_ens_data = {}
# for i, seed in tqdm.tqdm(enumerate(cfg_inference.seeds)):

i = 2
seed = 777
for model in cfg_inference.models:
    print(seed, model)
    version=cfg_inference[model][1].versions[i]
    ckpt_dir = glob.glob(os.path.join(cfg_inference.path, f"{model}_multi_{cfg_inference[model][0].upsampling}_{seed}/logs/version_{version}/checkpoints/"+"epoch_*.*"))
    cfg_train["model"].upsampling = cfg_inference[model][0].upsampling
    cfg_train.model.architecture = model
    cfg_train.training.checkpoint = ckpt_dir[-1]

    model_module, dm = main(cfg_train);
    model_module.eval().to('cuda:5')
    dm.setup()

    y_pred_train = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dm.predict_dataloader()):
            x, _, _, _ = batch
            y_pred_train.append(model_module(x.to('cuda:5'))[0].to('cpu'))

    # y_pred_val = []
    # with torch.no_grad():
    #     for batch in dm.val_dataloader():
    #         x, _, _, _ = batch
    #         y_pred_val.append(model_module(x.to('cuda:5'))[0].to('cpu'))

    # y_pred_test = []
    # with torch.no_grad():
    #     for batch in dm.test_dataloader():
    #         x, _, _, _ = batch
    #         y_pred_test.append(model_module(x.to('cuda:5'))[0].to('cpu'))
    
    train_ens_data[model + '_' + str(seed)] = torch.stack(y_pred_train)
    # val_ens_data[model + '_' + str(seed)] = torch.stack(y_pred_val)
    # test_ens_data[model + '_' + str(seed)] = torch.stack(y_pred_test)

print('saving train')
with open('train_ens_data_777.pkl', 'wb') as f:
    pickle.dump(train_ens_data, f)
print('done')

# print('saving val')
# with open('val_ens_data.pkl', 'wb') as f:
#     pickle.dump(val_ens_data, f)
# print('done')

# print('saving test')
# with open('test_ens_data.pkl', 'wb') as f:
#     pickle.dump(test_ens_data, f)
# print('done')
