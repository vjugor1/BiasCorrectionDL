# Standard library
from typing import Callable, List, Optional, Tuple, Union

# Local application
from ..data.processing.era5_constants import CONSTANTS

# Third party
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
import pytorch_lightning as pl


class LitModule(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: LRScheduler,
        train_loss: Callable,
        val_loss: List[Callable],
        test_loss: List[Callable],
        train_target_transform: Optional[Callable] = None,
        val_target_transforms: Optional[List[Union[Callable, None]]] = None,
        test_target_transforms: Optional[List[Union[Callable, None]]] = None,
    ):
        super().__init__()
        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.test_loss = test_loss
        self.train_target_transform = train_target_transform
        if val_target_transforms is not None:
            if len(val_loss) != len(val_target_transforms):
                raise RuntimeError(
                    "If 'val_target_transforms' is not None, its length must"
                    " match that of 'val_loss'. 'None' can be passed for"
                    " losses which do not require transformation."
                )
        self.val_target_transforms = val_target_transforms
        if test_target_transforms is not None:
            if len(test_loss) != len(test_target_transforms):
                raise RuntimeError(
                    "If 'test_target_transforms' is not None, its length must"
                    " match that of 'test_loss'. 'None' can be passed for "
                    " losses which do not rqeuire transformation."
                )
        self.test_target_transforms = test_target_transforms
        self.mode = "direct"

    def set_mode(self, mode):
        self.mode = mode

    def set_n_iters(self, iters):
        self.n_iters = iters

    def replace_constant(self, y, yhat, out_variables):
        for i in range(yhat.shape[1]):
            # if constant replace with ground-truth value
            if out_variables[i] in CONSTANTS:
                yhat[:, i] = y[:, i]
        return yhat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y, in_variables, out_variables = batch
        yhat = self(x).to(device=y.device)
        yhat = self.replace_constant(y, yhat, out_variables)
        if self.train_target_transform:
            yhat = self.train_target_transform(yhat)
            y = self.train_target_transform(y)
        losses = self.train_loss(yhat, y)
        loss_name = getattr(self.train_loss, "name", "loss")
        loss_dict = {}
        if losses.dim() == 0:  # aggregate loss only
            loss = losses
            loss_dict[f"train/{loss_name}:aggregate"] = loss
        else:  # per channel + aggregate
            for var_name, loss in zip(out_variables, losses):
                loss_dict[f"train/{loss_name}:{var_name}"] = loss
            loss = losses[-1]
            loss_dict[f"train/{loss_name}:aggregate"] = loss
        self.log_dict(
            loss_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=len(batch[0]),
        )
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
        batch_idx: int,
    ):
        self.evaluate(batch, "val")

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
        batch_idx: int,
    ):
        if self.mode == "direct":
            self.evaluate(batch, "test")
        if self.mode == "iter":
            self.evaluate_iter(batch, self.n_iters, "test")

    def evaluate(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]], stage: str
    ):
        x, y, in_variables, out_variables = batch
        yhat = self(x).to(device=y.device)
        yhat = self.replace_constant(y, yhat, out_variables)
        if stage == "val":
            loss_fns = self.val_loss
            transforms = self.val_target_transforms
        elif stage == "test":
            loss_fns = self.test_loss
            transforms = self.test_target_transforms
        else:
            raise RuntimeError("Invalid evaluation stage")
        loss_dict = {}
        for i, loss_fn in enumerate(loss_fns):
            # Apply the corresponding transformation if available
            if transforms is not None and transforms[i] is not None:
                yhat_transformed = transforms[i](yhat)
                y_transformed = transforms[i](y)
            else:
                yhat_transformed = yhat
                y_transformed = y
            
            # Calculate the losses
            losses = loss_fn(yhat_transformed, y_transformed)
            loss_name = getattr(loss_fn, "name", f"loss_{i}")

            # Check if the losses are aggregated or per channel
            if losses.dim() == 0:  # Aggregate loss
                loss_dict[f"{stage}/{loss_name}:aggregate"] = losses
            else:  # Per channel + aggregate loss
                for var_name, loss in zip(out_variables, losses):
                    loss_dict[f"{stage}/{loss_name}:{var_name}"] = loss
                # Add the aggregate loss
                loss_dict[f"{stage}/{loss_name}:aggregate"] = losses[-1]
        self.log_dict(
        loss_dict,
        prog_bar=True,
        on_step=False,
        on_epoch=True,
        sync_dist=True,
        batch_size=len(batch[0]),
        )
        return loss_dict

    def evaluate_iter(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
        n_iters: int,
        stage: str,
    ):
        x, y, in_variables, out_variables = batch

        x_iter = x
        for _ in range(n_iters):
            yhat_iter = self(x_iter).to(device=x_iter.device)
            yhat_iter = self.replace_constant(y, yhat_iter, out_variables)
            x_iter = x_iter[:, 1:]
            x_iter = torch.cat((x_iter, yhat_iter.unsqueeze(1)), dim=1)
        yhat = yhat_iter

        if stage == "val":
            loss_fns = self.val_loss
            transforms = self.val_target_transforms
        elif stage == "test":
            loss_fns = self.test_loss
            transforms = self.test_target_transforms
        else:
            raise RuntimeError("Invalid evaluation stage")
        loss_dict = {}
        for i, lf in enumerate(loss_fns):
            if transforms is not None and transforms[i] is not None:
                yhat_t = transforms[i](yhat)
                y_t = transforms[i](y)
            else:
                yhat_t = yhat
                y_t = y
            losses = lf(yhat_t, y_t)
            loss_name = getattr(lf, "name", f"loss_{i}")
            if losses.dim() == 0:  # aggregate loss
                loss_dict[f"{stage}/{loss_name}:aggregate"] = losses
            else:  # per channel + aggregate
                for var_name, loss in zip(out_variables, losses):
                    name = f"{stage}/{loss_name}:{var_name}"
                    loss_dict[name] = loss
                loss_dict[f"{stage}/{loss_name}:aggregate"] = losses[-1]
        self.log_dict(
            loss_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(batch[0]),
        )
        return loss_dict

    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return self.optimizer
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler = {
                "scheduler": self.lr_scheduler,
                "monitor": self.trainer.favorite_metric,
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            }
        else:
            scheduler = self.lr_scheduler
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler}

class DiffusionLitModule(LitModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: LRScheduler,
        train_loss: Callable,
        val_loss: List[Callable],
        test_loss: List[Callable],
        train_target_transform: Optional[Callable] = None,
        val_target_transforms: Optional[List[Union[Callable, None]]] = None,
        test_target_transforms: Optional[List[Union[Callable, None]]] = None,
    ):
        super().__init__(net, optimizer, lr_scheduler,
                         train_loss, val_loss, test_loss,
                         train_target_transform, val_target_transforms,
                         test_target_transforms)
        #upscaler in net
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_lr = x
        x_lr_up = self.net.upscaler(x_lr[...,:self.net.rrdb.conv_last.out_channels,:, :]) #self.rrdb.conv_last.out_channels == len(out_variables)
        img_out, *_ = self.net.sample(x_lr, x_lr_up) 
        return img_out

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
    ) -> torch.Tensor:
        x, y, in_variables, out_variables = batch
        
        img_hr = y
        img_lr = x
        img_lr_up = self.net.upscaler(img_lr[...,:len(out_variables),:,:]) #extracting hr channels

        x = img_hr
        b, *_, device = *x.shape, x.device
        t = torch.randint(0, self.net.num_timesteps, (b,), device=device).long() 
        if self.net.use_rrdb:
            if self.net.fix_rrdb:
                self.net.rrdb.eval()
                with torch.no_grad():
                    rrdb_out, cond = self.net.rrdb(img_lr, True)
            else:
                rrdb_out, cond = self.net.rrdb(img_lr, True)
        else:
            rrdb_out = img_lr_up
            cond = img_lr
        x = self.net.img2res(x, img_lr_up)

        x_start = x
        noise = torch.randn_like(x_start)
        x_tp1_gt = self.net.q_sample(x_start=x_start, t=t, noise=noise)
        noise_pred = self.net.denoise_fn(x_tp1_gt, t, cond, img_lr_up)

        
        loss = self.train_loss(noise_pred, noise)
        # if self.net.loss_type == 'l1':
        #     loss = (noise - noise_pred).abs().mean()
        # elif self.net.loss_type == 'l2':
        #     loss = F.mse_loss(noise, noise_pred)
        # elif self.net.loss_type == 'ssim':
        #     loss = (noise - noise_pred).abs().mean()
        #     loss = loss + (1 - self.ssim_loss(noise, noise_pred))
        # else:
        #     raise NotImplementedError()
        loss_dict = {'train/q': loss}
        if not self.net.fix_rrdb:
            if self.net.aux_l1_loss:
                loss_dict['train/aux_l1'] = F.l1_loss(rrdb_out, img_hr)
                loss += loss_dict['train/aux_l1']
            if self.net.aux_ssim_loss:
                loss_dict['train/aux_ssim'] = 1 - self.net.ssim_loss(rrdb_out, img_hr)
                loss += loss_dict['train/aux_ssim']
            # if hparams['aux_percep_loss']:
            #     loss_dict['aux_percep'] = self.percep_loss_fn[0](img_hr, rrdb_out)
        
        self.log_dict(
            loss_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=x.shape[0],
        )
        return loss

class YnetLitModule(LitModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: LRScheduler,
        train_loss: Callable,
        val_loss: List[Callable],
        test_loss: List[Callable],
        train_target_transform: Optional[Callable] = None,
        val_target_transforms: Optional[List[Union[Callable, None]]] = None,
        test_target_transforms: Optional[List[Union[Callable, None]]] = None,
        x_aux: Optional[List[Union[torch.Tensor, None]]] = None,
    ):
        super().__init__(net, optimizer, lr_scheduler,
                         train_loss, val_loss, test_loss,
                         train_target_transform, val_target_transforms,
                         test_target_transforms)
        self.x_aux = x_aux

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.x_aux is not None:
            x_aux_expanded = self.x_aux.to(x.dtype).to(x.device)
            x_aux_expanded = x_aux_expanded.unsqueeze(0).expand(x.size(0), *self.x_aux.size())
            return self.net(x, x_aux_expanded)
        return self.net(x)

class DeepSDLitModule(LitModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: LRScheduler,
        train_loss: Callable,
        val_loss: List[Callable],
        test_loss: List[Callable],
        train_target_transform: Optional[Callable] = None,
        val_target_transforms: Optional[List[Union[Callable, None]]] = None,
        test_target_transforms: Optional[List[Union[Callable, None]]] = None,
        elevation: Optional[List[Union[torch.Tensor, None]]] = None,
    ):
        super().__init__(net, optimizer, lr_scheduler,
                         train_loss, val_loss, test_loss,
                         train_target_transform, val_target_transforms,
                         test_target_transforms)
        self.elevation = elevation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.elevation is not None:
            # Ensure all elevation tensors are on the same device as x
            elevation_on_device = [e.to(x.device) for e in self.elevation]
            elevation_expanded = [e.unsqueeze(0).expand(x.size(0), *e.size()) for e in elevation_on_device]
            return self.net(x, elevation_expanded)
        return self.net(x)


class GANLitModule(LitModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
        lr_scheduler: tuple[LRScheduler, LRScheduler],
        train_loss: Callable,
        val_loss: List[Callable] = None,
        test_loss: List[Callable] = None,
        train_target_transform: Optional[Callable] = None,
        val_target_transforms: Optional[List[Union[Callable, None]]] = None,
        test_target_transforms: Optional[List[Union[Callable, None]]] = None,
        elevation: Optional[List[Union[torch.Tensor, None]]] = None,

    ):
        super().__init__(net, optimizer, lr_scheduler,
                         train_loss, val_loss, test_loss,
                         train_target_transform,
                         val_target_transforms,
                         test_target_transforms)
        
        self.elevation = elevation
        # Use this to perform the optimization in the training step
        self.automatic_optimization = False
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.elevation is not None:
            return self.net.generator(x, elevation=self.elevation.to(x.device))
        return self.net.generator(x) 

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], List[str]],
    ) -> torch.Tensor:
        x, y, _, _ = batch
        optimizerG, optimizerD = self.optimizers()
        lossG, lossD = self.train_loss
        
        # Fake:0, valid:1
        valid = torch.ones(x.size(0), 1).type_as(x)
        fake = torch.zeros(x.size(0), 1).type_as(x)

        # train generator
        self.toggle_optimizer(optimizerG)
        generated = self(x)
        advs_loss = lossD(self.net.discriminator(generated), valid)
        cont_loss = lossG(generated, y) # content loss
        g_loss = self.net.wmse * cont_loss + advs_loss # but usually take 1e-3 coeff for advs_loss
        
        self.manual_backward(g_loss)
        optimizerG.step()
        optimizerG.zero_grad()
        self.untoggle_optimizer(optimizerG)

        # train discriminator
        self.toggle_optimizer(optimizerD)
        real_loss = lossD(self.net.discriminator(y), valid)
        fake_loss = lossD(self.net.discriminator(self(x).detach()), fake)
        d_loss = (real_loss + fake_loss) / 2 
        
        self.manual_backward(d_loss)
        optimizerD.step()
        optimizerD.zero_grad()
        self.untoggle_optimizer(optimizerD)

        losses= {'advLoss': advs_loss, 'contLoss': cont_loss,
                 'lossG': g_loss, 'lossD': d_loss}
        self.log_dict(
                losses,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                batch_size=x.shape[0],
                )
    
    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None:
            if not isinstance(sch, list):
                sch = [sch]
            for i, scheduler in enumerate(sch):
                if i == 0:
                    # Generator
                    metric = "lossG"
                else:
                    # Discriminator
                    metric = "lossD"
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(self.trainer.callback_metrics[metric])
                else:
                    scheduler.step()

    def configure_optimizers(self):
        optimizerG, optimizerD = self.optimizer
        if self.lr_scheduler is None:
            return [optimizerG, optimizerD], []
        else:
            schedulerG, schedulerD = self.lr_scheduler
        return [optimizerG, optimizerD], [schedulerG, schedulerD]
