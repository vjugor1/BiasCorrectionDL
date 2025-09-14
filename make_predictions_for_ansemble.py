import glob
import os
import torch
import tqdm
import pickle

from hydra import compose, initialize
from cmip6_cmip6_dl_base_model import main

FACTOR = 2
CONFIG_NAME_TRAIN = "cmip6-cmip6"
CONFIG_NAME_INFERENCE = f"cmip6-cmip6_factor{FACTOR}_ens"

with initialize(version_base=None, config_path="configs/inference"):
    cfg_inference = compose(config_name=CONFIG_NAME_INFERENCE)

with initialize(version_base=None, config_path="configs/train"):
    cfg_train = compose(config_name=CONFIG_NAME_TRAIN)

train_ens_data = {}
val_ens_data = {}
test_ens_data = {}
i=1
models = cfg_inference.models_base
seed = cfg_train.training.seed
device = 'cuda:4'

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

    # y_pred_train = []
    # with torch.inference_mode():
    #     for batch in tqdm.tqdm(dm.predict_dataloader()):
    #         x, _, _, _ = batch
    #         y_pred_train.append(model_module(x.to(device, non_blocking=True)))

    # y_pred_val = []
    # with torch.inference_mode():
    #     for batch in dm.val_dataloader():
    #         x, _, _, _ = batch
    #         y_pred_val.append(model_module(x.to(device, non_blocking=True)))

    # y_pred_test = []
    # with torch.inference_mode():
    #     for batch in dm.test_dataloader():
    #         x, _, _, _ = batch
    #         y_pred_test.append(model_module(x.to(device, non_blocking=True)))

    # train_ens_data[model + '_' + str(seed)] = torch.cat(y_pred_train, dim=0) #.to('cpu')
    # val_ens_data[model + '_' + str(seed)] = torch.cat(y_pred_val, dim=0) #. to('cpu')
    # test_ens_data[model + '_' + str(seed)] = torch.cat(y_pred_test, dim=0) #.to('cpu')

    # print('saving train')
    # torch.save(train_ens_data, f'train_ens_data_factor_{FACTOR}_{model}_{seed}.pt')
    # # with open(f'train_ens_data_factor_4_{model}_{seed}.pkl', 'wb') as f:
    # #     pickle.dump(train_ens_data, f)
    # print('done')

    # print('saving val')
    # torch.save(train_ens_data, f'val_ens_data_factor_{FACTOR}_{model}_{seed}.pt')
    # # with open(f'val_ens_data_factor_4_{model}_{seed}.pkl', 'wb') as f:
    # #     pickle.dump(val_ens_data, f)
    # print('done')

    # print('saving test')
    # torch.save(train_ens_data, f'test_ens_data_factor_{FACTOR}_{model}_{seed}.pt')
    # # with open(f'test_ens_data_factor_4_{model}_{seed}.pkl', 'wb') as f:
    # #     pickle.dump(test_ens_data, f)
    # print('done')
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