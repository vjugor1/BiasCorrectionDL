import sys,os
sys.path.append(os.getcwd())
import logging
import numpy as np
import pytorch_lightning as pl
import torchvision  
from torch.utils.data import DataLoader, Dataset
import torch
from omegaconf import DictConfig
from src.regression.data_load import DataPreLoader
from src.utils.norm_values import mean_channels_cmip6, std_channels_cmip6


class WindDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg      
        self.DPL = DataPreLoader(cfg)
        
        if self.cfg.train.normalize:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Normalize(mean=mean_channels_cmip6, std=std_channels_cmip6),
                ]
            )
        else: self.transform = None
        
    def setup(self, stage=None):
        if self.cfg.train.use_elevation:
            logging.info("Using elevation data")
            # DatasetClass = XarrayDatasetElev
            raise NotImplementedError
        else:
            logging.info("Not using elevation data")
            DatasetClass = XarrayDataset

        if stage == "fit" or stage is None:
            self.dataset_train = DatasetClass(DPL=self.DPL, test=False)
            self.dataset_val = DatasetClass(DPL=self.DPL, test=True)
        if stage == "test":
            self.dataset_test = DatasetClass(DPL=self.DPL, test=True)


    def train_dataloader(self):
        return DataLoader(dataset=self.dataset_train,
                          batch_size=self.cfg.train.batch_size,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.dataset_val,
                          batch_size=self.cfg.train.batch_size,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.dataset_test,
                          batch_size=self.cfg.train.batch_size,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          )


class XarrayDataset(Dataset):
    def __init__(self, DPL, test=False, dtype=torch.float32):
        self.cfg = DPL.cfg
        self.dataset_torch = DPL.dataset_torch
        if test:
            self.data_idxs = DPL.test_data_idxs
            logging.info("Test dataloader init")
        else:
            self.data_idxs = DPL.train_data_idxs
            logging.info("Train dataloader init")
        self.dtype = dtype

        logging.info(f"Sample shape is {self.get_sample_shape(10)}")

    def get_sample_shape(self, idx): 
        lat_index, lon_index, time_index, *y = self.data_idxs[:, idx]
        lat_index = int(lat_index)
        lon_index = int(lon_index)
        time_index = int(time_index)
        X = self.dataset_torch[:,
                               slice(time_index - self.cfg.time_window//2, time_index + self.cfg.time_window//2 + 1),
                               slice(lat_index - self.cfg.half_side_size, lat_index + self.cfg.half_side_size + 1),
                               slice(lon_index - self.cfg.half_side_size, lon_index + self.cfg.half_side_size + 1),
                               ]
        return X.shape

    def __len__(self):
        return self.data_idxs.shape[1]

    def __getitem__(self, idx):
        lat_index, lon_index, time_index, time_pos, time_pos_m, lat_pos, lon_pos, *y = self.data_idxs[:, idx]
        lat_index = int(lat_index)
        lon_index = int(lon_index)
        time_index = int(time_index)
        X = self.dataset_torch[:,
                               slice(time_index - self.cfg.time_window//2, time_index + self.cfg.time_window//2 + 1),
                               slice(lat_index - self.cfg.half_side_size, lat_index + self.cfg.half_side_size + 1),
                               slice(lon_index - self.cfg.half_side_size, lon_index + self.cfg.half_side_size + 1),
                               ]
        # assert X.shape[-1] == 95, f"idxs {lat_index, lon_index, time_index}"
        pos = torch.tensor([time_pos, time_pos_m, lat_pos, lon_pos], dtype=self.dtype)
        pos = pos.expand(self.cfg.time_window, 4)
        y = torch.tensor(y, dtype=self.dtype)
        return [X, pos], y
    

if __name__ == '__main__':
    pass