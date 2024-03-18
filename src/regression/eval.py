import sys,os
sys.path.append(os.getcwd())
import warnings
import torch
import logging
from tqdm import tqdm
from src.regression.data_load import make_padding
from src.regression.models.pl_module import WindNetPL
import hydra
from omegaconf import DictConfig
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch.nn as nn
from collections import OrderedDict


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

def plot_prediction(cfg: DictConfig, predictions_df: np.ndarray, days: int) -> None:
    given_days = predictions_df['date'].unique()
    logging.info(f"Plot in coords:")
    logging.info(f"Lat {predictions_df['lat'].min()}-{predictions_df['lat'].max()}")
    logging.info(f"Lon {predictions_df['lon'].min()}-{predictions_df['lon'].max()}")
    for day in given_days[:days]:
        logging.info(f"Day: {day}")
        current_data = predictions_df[predictions_df['date'] == day]
        pivot_table = current_data.pivot(index='lat', columns='lon', values='prediction')    
        image_array = pivot_table.values
        fig, ax = plt.subplots(figsize=(12, 12))
        plt.title(f'Max wind speed prediction for {pd.to_datetime(day).date()}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        img = ax.imshow(image_array, interpolation='nearest', extent=[predictions_df['lon'].min(),predictions_df['lon'].max(),
                                                                      predictions_df['lat'].max(), predictions_df['lat'].min()])
        plt.gca().invert_yaxis()
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        fig.colorbar(img, cax=cax)
        savepath = os.path.join(*cfg.eval.path_to_predictions.split('/')[:-1], f'{cfg.target_type}_{pd.to_datetime(day).date()}.png')
        plt.savefig(savepath, dpi=200) 
        logging.info(f"Plot saved to {savepath}")
        plt.clf()
        ax.cla()

    
class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.prepare_data()
        self.total_index = list(itertools.product(self.time_indexes, self.lat_indexes, self.lon_indexes))
        logging.info(f"Coordinates to be predicted")
        logging.info(f"Lat: {min(self.lat_coords)} - {max(self.lat_coords)}")
        logging.info(f"Lon: {min(self.lon_coords)} - {max(self.lon_coords)}")
        logging.info(f"Time: {min(self.time_coords)} - {max(self.time_coords)}")

    def prepare_data(self):
        """Prepare data for inference"""
        self.load_dataset()
        self.limit_inference_time()
        self.var_data, self.shift = make_padding(self.var_data, self.cfg.half_side_size)
        logging.info(f"Padded data shape: {self.var_data.shape}")
        self.limit_inference_space()
        logging.info(f"Final data shape: {self.var_data.shape}")

        assert len(self.lat_indexes) == len(self.lat_coords), f" indexes is {len(self.lat_indexes)} while coords is {len(self.lat_coords)}"
        assert len(self.lon_indexes)== len(self.lon_coords), f" indexes is {len(self.lon_indexes)} while coords is  {len(self.lon_coords)}"
        assert len(self.time_indexes)== len(self.time_coords), f" indexes is {len(self.time_indexes)} while coords is {len(self.time_coords)}"

    def load_dataset(self):
        """Load climate data from given folder"""
        self.time_coords_full = np.load(os.path.join(self.cfg.eval.data_dir, 'time.npy')).astype('datetime64[D]')
        self.lat_coords_full = np.load(os.path.join(self.cfg.eval.data_dir, 'lat.npy'))
        self.lon_coords_full = np.load(os.path.join(self.cfg.eval.data_dir, 'lon.npy'))
        self.var_data = np.empty((len(self.cfg.process.variables), len(self.time_coords_full), len(self.lat_coords_full), len(self.lon_coords_full)), dtype=np.float16)
        
        logging.info(f"Using vars: {self.cfg.process.variables}") 
        for i, var in enumerate(self.cfg.process.variables):
            self.var_data[i] = np.load(os.path.join(self.cfg.eval.data_dir, var + f'_{self.cfg.process.precision}.npy'))

    def limit_inference_time(self):
        """Crop data by spatial coordinates"""
        time_start = pd.to_datetime(self.cfg.start_date)
        time_end = pd.to_datetime(self.cfg.end_date)
        self.time_min_idx = np.searchsorted(self.time_coords_full, time_start.date())
        self.time_max_idx = np.searchsorted(self.time_coords_full, time_end.date())

        self.time_coords = self.time_coords_full[self.time_min_idx:self.time_max_idx]
        self.time_indexes = list(range(self.time_min_idx + self.cfg.time_window//2, self.time_max_idx + self.cfg.time_window//2)) 
        self.lat_indexes = list(range(self.cfg.half_side_size, len(self.lat_coords_full) + self.cfg.half_side_size))
        self.lon_indexes = list(range(self.cfg.half_side_size, len(self.lon_coords_full) + self.cfg.half_side_size))
        
        self.var_data = self.var_data[:, self.time_min_idx - self.cfg.time_window//2:self.time_max_idx + self.cfg.time_window//2, :, :]
        self.time_indexes = [t_idx - self.time_min_idx for t_idx in self.time_indexes]
        if isinstance(self.time_coords, list) and isinstance(self.time_indexes, list):
            assert len(self.time_indexes) == len(self.time_coords), f"{len(self.time_indexes)} {len(self.time_coords)}"
            assert self.var_data.shape[1] == len(self.time_coords), f"{len(self.var_data.shape[1])} {len(self.time_coords)}"
        logging.info(f"Time bounded data shape: {self.var_data.shape}")


    def limit_inference_space(self):
        """Crop data by spatial coordinates"""
        #find bounding indexes 
        self.lat_min_idx = np.searchsorted(self.lat_coords_full, self.cfg.eval.lat_min)
        self.lat_max_idx = np.searchsorted(self.lat_coords_full, self.cfg.eval.lat_max)
        self.lon_min_idx = np.searchsorted(self.lon_coords_full, self.cfg.eval.lon_min)
        self.lon_max_idx = np.searchsorted(self.lon_coords_full, self.cfg.eval.lon_max)
        logging.info(f"lon_min_idx: {self.lon_min_idx}")
        logging.info(f"lon_max_idx: {self.lon_max_idx}")
        self.lat_coords = self.lat_coords_full[self.lat_min_idx:self.lat_max_idx]
        self.lon_coords = self.lon_coords_full[self.lon_min_idx:self.lon_max_idx]
        #list of indexes with padding shift
        self.lat_indexes = list(range(self.lat_min_idx + self.shift[0], self.lat_max_idx + self.shift[0]))
        self.lon_indexes = list(range(self.lon_min_idx + self.shift[1], self.lon_max_idx + self.shift[1]))
        self.var_data = self.var_data[
                                      :,
                                      :,
                                      self.lat_min_idx: self.lat_max_idx+2*self.cfg.half_side_size,
                                      self.lon_min_idx: self.lon_max_idx+2*self.cfg.half_side_size
                                      ]
        self.lat_indexes = [l_idx - self.lat_min_idx for l_idx in self.lat_indexes]
        self.lon_indexes = [l_idx - self.lon_min_idx for l_idx in self.lon_indexes]
        assert len(self.lat_indexes) == len(self.lat_coords), f"{len(self.lat_indexes)} {len(self.lat_coords)}"
        assert len(self.lon_indexes)== len(self.lon_coords), f"{len(self.lon_indexes)} {len(self.lon_coords)}"
        assert len(self.time_indexes)== len(self.time_coords), f"{len(self.time_indexes)} {len(self.time_coords)}"

    def __len__(self):
        return len(self.total_index)

    def __getitem__(self, idx):
        t, lat, lon  = self.total_index[idx]
        item = self.var_data[:,
                            slice(t - self.cfg.time_window//2, t + self.cfg.time_window//2 + 1),
                            slice(lat - self.cfg.half_side_size, lat + self.cfg.half_side_size + 1),
                            slice(lon - self.cfg.half_side_size, lon + self.cfg.half_side_size + 1),
                            ]
        
        time_pos_month = self.time_coords[t- self.cfg.time_window//2].astype(object).month
        time_pos_day =  self.time_coords[t- self.cfg.time_window//2].astype(object).day
        time_pos_day = (time_pos_month * 30.5 + time_pos_day)/365
        time_pos_month = time_pos_month/12
        lat_pos = self.lat_coords[lat - self.shift[0]]/90
        lon_pos = self.lon_coords[lon - self.shift[1]]/180
        pos = torch.tensor([time_pos_day, time_pos_month, lat_pos, lon_pos], dtype=torch.float32)
        pos = pos.expand(self.cfg.time_window, 4)
        item = [torch.from_numpy(item).to(torch.float32), pos]
        coord = np.array([t, lat, lon])
        return item, coord
    
def load_model_fixed(model, path):
    ckpt = torch.load(path)
    state_dict = torch.load(path)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if '._orig_mod' in k:
            name = k.replace('._orig_mod', '') # remove `._orig_mod`
        else:
            name = k
        if name in new_state_dict:
            raise ValueError
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model


def predict(model, dataset, use_elevation, batch_size=1, distributed=False, device_num=0, num_workers=0):      
    if torch.cuda.is_available():
        torch.cuda.set_device(device_num)
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")
    logging.info(f'Using {device}')
    if distributed:
        model = nn.DataParallel(model)
    else:
        model.to(device)
    number_of_points = len(dataset.lat_indexes) * len(dataset.lon_indexes) * len(dataset.time_indexes)
    logging.info((f"Number of points to predict: {number_of_points}"))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
    predictions_list = []
    coords_list = []  
    coords_indxs_list =  []
    num_items_to_predict = len(dataloader)
    logging.info(f"Number of batches to predict: {num_items_to_predict}")

    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_items_to_predict, desc="Inference"):
            data = batch[0]
            coords_idxs = batch[1]
            # logging.info(f'data {data[0].max()} {data[0].min()} {data[0].std()}')
            if not distributed:
                data = [t.to(device) for t in data]
            prediction = model(data)[:, 1]
            prediction = prediction.detach().cpu()
            prediction = prediction.numpy()
            # logging.info(f'preds {prediction.max()} {prediction.min()} {prediction.std()}')
            predictions_list = predictions_list + list(prediction)
            coords_indxs_list.append(coords_idxs)
    logging.info("Inference finished")

    for coords_idxs in coords_indxs_list:
        for t, lat, lon, in coords_idxs:
            coords = [dataset.lat_coords[lat - dataset.shift[0]],
                      dataset.lon_coords[lon - dataset.shift[1]],
                      dataset.time_coords[t - dataset.cfg.time_window//2]]
            coords_list.append(coords)

    print(len(predictions_list))
    predictions_array = np.array(predictions_list)
    print(predictions_array.shape)
    logging.info(f"Predictions max value: {predictions_array.max()}, min value: {predictions_array.min()}")
    logging.info(f"Predictions mean value: {predictions_array.mean()}, std value: {predictions_array.std()}")
    result_df = pd.DataFrame({"date": [item[2] for item in coords_list],
                              "lat": [item[0] for item in coords_list],
                              "lon": [item[1] for item in coords_list],
                              "prediction": predictions_array.flatten().astype(np.float64)
                           })
    return result_df


def eval(cfg: DictConfig) -> None:   
    model = WindNetPL(cfg=cfg, eval=True)     
    # checkpoint = torch.load(cfg.eval.path_to_checkpoint)
    # print(checkpoint['state_dict'].keys())
    model = load_model_fixed(model, cfg.eval.path_to_checkpoint).eval()
    model= torch.compile(model).eval()
    # model = model.load_from_checkpoint(cfg.eval.path_to_checkpoint, cfg=cfg).half().eval()

    dataset = EvalDataset(cfg)
    result_df = predict(model, dataset,
                        use_elevation=cfg.eval.use_elevation,
                        batch_size=cfg.eval.batch_size_test,
                        distributed=cfg.eval.distributed_test,
                        num_workers=cfg.eval.num_workers_eval)
                        
    os.makedirs(os.path.join(*cfg.eval.path_to_predictions.split('/')[:-1]), exist_ok=True)
    result_df.to_csv(cfg.eval.path_to_predictions, index=False)
    logging.info(f"Saved to {cfg.eval.path_to_predictions}")
    plot_prediction(cfg, result_df, 1)


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="cmip6_GhostWindNet27.yaml")
def main(cfg: DictConfig):    
    eval(cfg)

if __name__ == "__main__":      
    sys.argv.append('hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()