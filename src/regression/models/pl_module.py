import sys,os
sys.path.append(os.getcwd())
from typing import List, Any
import torch
import pytorch_lightning as pl
from collections import OrderedDict
import torchmetrics
from torchmetrics import MaxMetric, MeanMetric, MinMetric
from torch.functional import F
import torch.nn as nn
from src.regression.models.models import *
from src.utils.metrics import float_to_binary, float_to_score, get_outliers_s, get_outliers_p
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

class WindNetPL(pl.LightningModule):

    def __init__(self, cfg, run_dir=None, eval=False): 
        super().__init__()     
        self.cfg = cfg        
        self.run_dir = run_dir
        if cfg.model_name=="BaselineLinear":
             self.net = BaselineLinear()
        elif cfg.model_name=="BaselineQT":
            self.net = BaselineQT()
        elif cfg.model_name=="BaselineQW":
            self.net = BaselineQW()
        elif cfg.model_name=="GhostWindNet27":
            self.net = GhostWindNet27()
        else:
            raise NotImplementedError(f'Model {cfg.model_name} not found')     
        logging.info(f'Using {cfg.model_name} model')
        if not eval:
            self.scheduler_name = cfg.train.scheduler_name
            if cfg.train.optimizer_name=='AdamW':
                self.optimizer = torch.optim.AdamW
            elif cfg.train.optimizer_name=='RAdam':
                self.optimizer = torch.optim.RAdam
            elif cfg.train.optimizer_name=='SGD':
                self.optimizer = torch.optim.SGD
            else:
                raise NotImplementedError(f'Optimizer {cfg.train.optimizer_name} not found')
            
            if cfg.train.loss_name=='MSELoss':
                self.criterion = torch.nn.MSELoss()
            elif cfg.train.loss_name=='L1Loss':
                self.criterion = torch.nn.L1Loss()
            else:
                raise NotImplementedError(f'Criterion {cfg.train.loss_name} not found')
        
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_MAE_best = MinMetric()

        self.train_AP = torchmetrics.AveragePrecision(num_classes=1, task='binary')
        self.val_AP = torchmetrics.AveragePrecision(num_classes=1, task='binary')
        self.test_AP = torchmetrics.AveragePrecision(num_classes=1, task='binary')

        self.test_auroc = torchmetrics.AUROC(task="binary")

        self.train_MAE = torchmetrics.MeanAbsoluteError()
        self.val_MAE = torchmetrics.MeanAbsoluteError()
        self.test_MAE = torchmetrics.MeanAbsoluteError()
        self.train_MAE_full = torchmetrics.MeanAbsoluteError()
        self.val_MAE_full = torchmetrics.MeanAbsoluteError()
        self.test_MAE_full = torchmetrics.MeanAbsoluteError()

        self.val_precision = torchmetrics.Precision(num_classes=1, task='binary')
        self.val_recall = torchmetrics.Recall(num_classes=1, task='binary')
        self.test_precision = torchmetrics.Precision(num_classes=1, task='binary')
        self.test_recall = torchmetrics.Recall(num_classes=1, task='binary')

        self.train_MAE_OS = torchmetrics.MeanAbsoluteError() # MAE outliers based on station measure
        self.val_MAE_OS = torchmetrics.MeanAbsoluteError() 
        self.test_MAE_OS = torchmetrics.MeanAbsoluteError() 


    def forward(self, x):
        return self.net(x)

    def loss(self, y_hat, y):        
        return self.criterion(y_hat, y)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)
        self.val_MAE_best.reset()

    def model_step(self, batch):
        objs, target = batch
        # print(objs[0].shape)
        # print(objs[1].shape)
        predictions = self(objs).float()
        loss = self.loss(predictions, target.float())
        return loss, predictions, target    
    
    def training_step(self, batch, batch_idx):
        loss, predictions, target = self.model_step(batch)
        self.train_loss(loss)
        self.train_MAE(predictions[:, 0], target[:, 0])
        self.train_MAE_full(predictions, target)
        self.train_MAE_OS(*get_outliers_s(predictions[:, 0], target[:, 0] , thresh=self.cfg.train.target_threshold ))
        self.train_AP(float_to_score(predictions[:, 0], thresh=self.cfg.train.target_threshold ),
                      float_to_binary(target[:, 0], thresh=self.cfg.train.target_threshold))

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True)
        self.log("train/MAE", self.train_MAE, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/MAE_full", self.train_MAE_full, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/MAE_OS", self.train_MAE_OS, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/AP", self.train_AP, on_step=True, on_epoch=True, prog_bar=True)
        #if batch_idx%100==0:
        #    self.logger.experiment.log({"train/target": target[:, 0], "train/prediction": predictions[:, 0]})
        #    self.logger.experiment.log({"train/target_50": target[:, 3], "train/prediction_50": predictions[:, 3]})

        output = OrderedDict(
            {
                "loss": loss,
                "preds": predictions,
                "target": target,
            }
        )
        return output
            

    def validation_step(self, batch, batch_idx):
        loss, predictions, target = self.model_step(batch)

        self.val_loss(loss)
        self.val_MAE(predictions[:, 0], target[:, 0])
        self.val_MAE_full(predictions, target)

        self.val_AP(float_to_score(predictions[:, 0], thresh=self.cfg.train.target_threshold),
                     float_to_binary(target[:, 0], thresh=self.cfg.train.target_threshold))
        self.val_precision(float_to_binary(predictions[:, 0], thresh=self.cfg.train.target_threshold),
                           float_to_binary(target[:, 0], thresh=self.cfg.train.target_threshold))
        self.val_recall(float_to_binary(predictions[:, 0], thresh=self.cfg.train.target_threshold),
                        float_to_binary(target[:, 0], thresh=self.cfg.train.target_threshold))        
        self.val_MAE_OS(*get_outliers_s(predictions[:, 0], target[:, 0], thresh=self.cfg.train.target_threshold))

        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/MAE", self.val_MAE, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/MAE_full", self.val_MAE_full, on_step=True, on_epoch=True, prog_bar=False)
        self.log("val/MAE_OS", self.val_MAE_OS, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/AP", self.val_AP, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True)

        if batch_idx%100==0:
            self.logger.experiment.log({"val/target_96": target[:, 0], "val/prediction_96": predictions[:, 0]})
            self.logger.experiment.log({"val/target_50": target[:, 3], "val/prediction_50": predictions[:, 3]})
            self.logger.experiment.log({"val/target_05": target[:, 6], "val/prediction_05": predictions[:, 6]})

        output = OrderedDict(
            {
                "loss": loss,
                "preds": predictions,
                "target": target,
            }
        )
        return output
    

    def on_validation_epoch_end(self):
        MAE = self.val_MAE.compute()
        self.val_MAE_best(MAE)
        self.log("val/MAE_best", self.val_MAE_best.compute(), prog_bar=False)


    def test_step(self, batch, batch_idx):
        loss, predictions, target = self.model_step(batch)

        binary_target = float_to_binary(target[:, 0], thresh=self.cfg.train.target_threshold)
        score_preds = float_to_score(predictions[:, 0], thresh=self.cfg.train.target_threshold)
        binary_preds = float_to_binary(predictions[:, 0], thresh=self.cfg.train.target_threshold)

        self.test_loss(loss)
        self.test_MAE(predictions[:, 0], target[:, 0])
        self.test_MAE_full(predictions, target)

        self.test_MAE_OS(*get_outliers_s(predictions[:, 0], target[:, 0], thresh=self.cfg.train.target_threshold))
        self.test_AP(score_preds, binary_target)
        self.test_precision(binary_preds,binary_target)
        self.test_recall(binary_preds, binary_target)
        self.test_auroc(score_preds, binary_target)
        
        self.log("test/loss", self.test_loss, prog_bar=True)
        self.log("test/MAE", self.test_MAE, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/MAE_full", self.test_MAE_full, on_step=True, on_epoch=True, prog_bar=False)

        self.log("test/MAE_OS", self.test_MAE_OS, on_step=True, on_epoch=True, prog_bar=False)
        self.log("test/AP", self.test_AP, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_precision, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/recall", self.test_recall, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/AUROC", self.test_auroc, on_epoch=True)

        if batch_idx%100==0:
            self.logger.experiment.log({"test/target_96": target[:, 0], "test/prediction_96": predictions[:, 0]})
            self.logger.experiment.log({"test/target_50": target[:, 3], "test/prediction_50": predictions[:, 3]})
            self.logger.experiment.log({"test/target_05": target[:, 6], "test/prediction_05": predictions[:, 6]})
        
        output = OrderedDict(
            {
                "loss": loss,
                "binary_preds": binary_preds,
                "score_preds": score_preds,
                "float_preds": predictions,
                "float_target": target,
                "binary_target": binary_target,
            }
        )
        self.test_outputs.append(output)

        return output
    
    def on_test_start(self):
        print(f'Plots will be saved to {self.run_dir}')
        os.makedirs(self.run_dir, exist_ok=True)
        self.test_outputs = []

    def on_test_epoch_end(self):
        preds = torch.stack([x["binary_preds"] for x in self.test_outputs]).to(dtype=torch.float32).cpu().flatten()
        target = torch.stack([x["binary_target"] for x in self.test_outputs]).to(dtype=torch.int32).cpu().flatten()
        preds_float = torch.stack([x["float_preds"] for x in self.test_outputs]).to(dtype=torch.float32).cpu().flatten()
        target_float = torch.stack([x["float_target"] for x in self.test_outputs]).to(dtype=torch.int32).cpu().flatten()

        thrs = [0, 3, 5, 8, 10, 12, 15, 17, 20, 23, 25, 27, 30]
        rmses = []
        for th in thrs:
            if len(torch.where(target_float >= th)[0])>0:
                tgt_th = target_float[torch.where(target_float >= th)[0]]
                pred_th = preds_float[torch.where(target_float >= th)[0]]
                rmses.append(np.squeeze(torch.sqrt(torch.mean((pred_th - tgt_th) ** 2)).numpy()))
        
        fig, ax = plt.subplots()
        ax.plot(thrs, rmses, color='purple')
        ax.set_ylabel('RMSE')
        ax.set_xlabel('Wind Speed (m/s)')
        fig.savefig(os.path.join(self.run_dir, 'RMSE_vs_target.png'))   # save the figure to file        

        precision, recall, thresholds = precision_recall_curve(target, preds)
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color='purple')
        ax.set_title('Precision-Recall Curve')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        fig.savefig(os.path.join(self.run_dir, 'PR_curve.png'))   # save the figure to file        


    def configure_optimizers(self):
        optimizer = self.optimizer(self.net.parameters(),
                                   lr=self.cfg.train.learning_rate,
                                   weight_decay=self.cfg.train.weight_decay)        
        if self.scheduler_name is not None:
            if self.scheduler_name == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.7, patience=300, verbose=True, interval="step", frequency=1)
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        "monitor": "train/loss",
                        "patience": 300,
                        "mode": "min",
                        "factor": 0.7,
                        "verbose": True,
                        'name': 'train/lr',
                        'scheduler': scheduler,
                        'interval': 'step', 
                        'frequency': 1,
                    }
                }
            
            elif self.scheduler_name == "OneCycleLR":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches)
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'name': 'train/lr',  # put lr inside train group in tensorboard
                        'scheduler': scheduler,
                        'interval': 'step', 
                        'frequency': 1,
                    }
                }
            elif self.scheduler_name == "LinearLR":
                scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                             start_factor=1.0, end_factor=0.2, 
                                                             total_iters=self.trainer.estimated_stepping_batches)
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'name': 'train/lr',
                        'scheduler': scheduler,
                        'interval': 'step', 
                        'frequency': 1,
                    }
                }

        else:
            return optimizer