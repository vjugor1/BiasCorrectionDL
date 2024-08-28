# Standard library
from typing import Any, Callable, Dict, Iterable, Optional, Union, Tuple
from functools import partial
import warnings
import numpy as np

# Local application
from .gis import prepare_ynet_climatology, prepare_deepsd_elevation, prepare_dcgan_elevation
from ..data import IterDataModule
from ..models import LitModule, DiffusionLitModule, DeepSDLitModule, YnetLitModule, GANLitModule, ESRGANLitModule, MODEL_REGISTRY
from ..models.hub import (
    Climatology,
    Interpolation,
    LinearRegression,
    Persistence,
    ResNet,
    Unet,
    UnetUpsampling,
    VisionTransformer,
    VisionTransformerSAM,
    GaussianDiffusion,
    YNet30,
    DeepSD,
    DCGAN,
    EDRN,
    ESRGAN,
    HAT
    
)
from ..models.lr_scheduler import LinearWarmupCosineAnnealingLR
from ..transforms import TRANSFORMS_REGISTRY
from ..metrics import MetricsMetaInfo, METRICS_REGISTRY

# Third party
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


def load_model_module(
    task: str,
    data_module,
    architecture: Optional[str] = None,
    model: Optional[Union[str, nn.Module]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    optim: Optional[Union[str, torch.optim.Optimizer]] = None,
    optim_kwargs: Optional[Dict[str, Any]] = None,
    sched: Optional[Union[str, LRScheduler]] = None,
    sched_kwargs: Optional[Dict[str, Any]] = None,
    upsampling: Optional[str] = None,
    train_loss: Optional[Union[str, Callable, Tuple]] = None,
    train_loss_kwargs: Optional[Dict[str, Any]] = None,
    val_loss: Optional[Iterable[Union[str, Callable]]] = None,
    test_loss: Optional[Iterable[Union[str, Callable]]] = None,
    train_target_transform: Optional[Union[str, Callable, Tuple]] = None,
    val_target_transform: Optional[Iterable[Union[str, Callable]]] = None,
    test_target_transform: Optional[Iterable[Union[str, Callable]]] = None,
    path_to_elevation: Optional[str] = "/app/data/elevation.nc",
):
    # Temporary fix, per this discussion:
    # https://github.com/aditya-grover/climate-learn/pull/100#discussion_r1192812343
    lat, lon = data_module.get_lat_lon()
    if lat is None and lon is None:
        raise RuntimeError("Data module has not been set up yet.")
    
    # Load the model
    if architecture is None and model is None:
        raise RuntimeError("Please specify 'architecture' or 'model'")
    elif architecture:
        print(f"Loading architecture: {architecture}")
        model, optimizer, lr_scheduler = load_architecture(
            task, data_module, architecture, upsampling, 
            optim_kwargs, model_kwargs, sched_kwargs
        )
    elif isinstance(model, str):
        print(f"Loading model: {model}")
        model_cls = MODEL_REGISTRY.get(model, None)
        if model_cls is None:
            raise NotImplementedError(
                f"{model} is not an implemented model. If you think it should be,"
                " please raise an issue at"
                " https://github.com/aditya-grover/climate-learn/issues."
            )
        model = model_cls(**model_kwargs)
    elif isinstance(model, nn.Module):
        print("Using custom network")
    else:
        raise TypeError("'model' must be str or nn.Module")
    
    # Load the optimizer
    if architecture is None and optim is None:
        raise RuntimeError("Please specify 'architecture' or 'optim'")
    elif architecture:
        print("Using optimizer associated with architecture")
    elif isinstance(optim, str):
        print(f"Loading optimizer {optim}")
        optimizer = load_optimizer(model, optim, optim_kwargs)
    elif isinstance(optim, torch.optim.Optimizer):
        optimizer = optim
        print("Using custom optimizer")
    else:
        raise TypeError("'optim' must be str or torch.optim.Optimizer")
    
    # Load the LR scheduler, if specified
    if architecture:
        print("Using learning rate scheduler associated with architecture")
    elif sched is None:
        lr_scheduler = None
    elif isinstance(sched, str):
        print(f"Loading learning rate scheduler: {sched}")
        lr_scheduler = load_lr_scheduler(sched, optimizer, sched_kwargs)
    elif isinstance(sched, LRScheduler) or isinstance(
        sched, torch.optim.lr_scheduler.ReduceLROnPlateau
    ):
        lr_scheduler = sched
        print("Using custom learning rate scheduler")
    else:
        raise TypeError(
            "'sched' must be str, None, or torch.optim.lr_scheduler._LRScheduler"
        )

    # Load training loss
    in_vars, out_vars = get_data_variables(data_module)
    lat, lon = data_module.get_lat_lon()
    if isinstance(train_loss, str):
        print(f"Loading training loss: {train_loss}")
        clim = get_climatology(data_module, "train")
        metainfo = MetricsMetaInfo(in_vars, out_vars, lat, lon, clim)
        if train_loss == 'perceptual':
            train_loss = load_loss(train_loss, True, metainfo, **train_loss_kwargs)
        else:
            train_loss = load_loss(train_loss, True, metainfo)
    elif isinstance(train_loss, tuple):
        print(f"Loading training loss: {train_loss}")
        clim = get_climatology(data_module, "train")
        metainfo = MetricsMetaInfo(in_vars, out_vars, lat, lon, clim)
        train_lossG = load_loss(train_loss[0], True, metainfo)
        train_lossD = load_loss(train_loss[1], True, metainfo)
        train_loss = (train_lossG, train_lossD)
    elif isinstance(train_loss, Callable):
        print("Using custom training loss")
    else:
        raise TypeError("'train_loss' must be str, tuple or Callable")

    # Load training transform
    if isinstance(train_target_transform, str):
        print(f"Loading training transform: {train_target_transform}")
        train_transform = load_transform(train_target_transform, data_module)
    elif isinstance(train_target_transform, tuple):
        print(f"Loading training transform: {train_target_transform}")
        train_transformG = load_transform(train_target_transform, data_module)
        train_transformD = load_transform(train_target_transform, data_module)
        train_transform = (train_transformG, train_transformD)
    elif isinstance(train_target_transform, Callable):
        print("Using custom training transform")
        train_transform = train_target_transform
    elif train_target_transform is None:
        print("No train transform")
        train_transform = train_target_transform
    else:
        raise TypeError("'train_target_transform' must be str, tuple, Callable, or None")
    
    # Load validation loss
    if not isinstance(val_loss, Iterable):
        raise TypeError("'val_loss' must be an iterable")
    val_losses = []
    for vl in val_loss:
        if isinstance(vl, str):
            clim = get_climatology(data_module, "val")
            metainfo = MetricsMetaInfo(in_vars, out_vars, lat, lon, clim)
            print(f"Loading validation loss: {vl}")
            val_losses.append(load_loss(vl, False, metainfo))
        elif isinstance(vl, Callable):
            print("Using custom validation loss")
            val_losses.append(vl)
        else:
            raise TypeError("each 'val_loss' must be str or Callable")
        
    # Load validation transform
    val_transforms = []
    if isinstance(val_target_transform, Iterable):
        for vt in val_target_transform:
            if isinstance(vt, str):
                print(f"Loading validation transform: {vt}")
                val_transforms.append(load_transform(vt, data_module))
            elif isinstance(vt, Callable):
                print("Using custom validation transform")
                val_transforms.append(vt)
            elif vt is None:
                print("No validation transform")
                val_transforms.append(None)
            else:
                raise TypeError("each 'val_transform' must be str, Callable, or None")
    elif val_target_transform is None:
        val_transforms = val_target_transform
    else:
        raise TypeError(
            "'val_target_transform' must be an iterable of strings/callables,"
            " or None"
        )
        
    # Load test loss
    if not isinstance(test_loss, Iterable):
        raise TypeError("'test_loss' must be an iterable")
    test_losses = []
    for tl in test_loss:
        if isinstance(tl, str):
            clim = get_climatology(data_module, "test")
            metainfo = MetricsMetaInfo(in_vars, out_vars, lat, lon, clim)
            print(f"Loading test loss: {tl}")
            test_losses.append(load_loss(tl, False, metainfo))
        elif isinstance(tl, Callable):
            print("Using custom testing loss")
            test_losses.append(tl)
        else:
            raise TypeError("each 'test_loss' must be str or Callable")
        
    # Load test transform
    test_transforms = []
    if isinstance(test_target_transform, Iterable):
        for tt in test_target_transform:
            if isinstance(tt, str):
                print(f"Loading test transform: {tt}")
                test_transforms.append(load_transform(tt, data_module))
            elif isinstance(tt, Callable):
                print("Using custom test transform")
                test_transforms.append(tt)
            elif tt is None:
                print("No test transform")
                test_transforms.append(None)
            else:
                raise TypeError("each 'test_transform' must be str, Callable, or None")
    elif test_target_transform is None:
        test_transforms = test_target_transform
    else:
        raise TypeError(
            "'test_target_transform' must be an iterable of strings/callables,"
            " or None"
        )
    # Instantiate Lightning Module
    if architecture == 'diffusion':
        model_module = DiffusionLitModule(
            model,
            optimizer,
            lr_scheduler,
            train_loss,
            val_losses,
            test_losses,
            train_transform,
            val_transforms,
            test_transforms,
        )
    elif architecture == "ynet":
        normalized_clim = prepare_ynet_climatology(data_module, path_to_elevation, out_vars)
        
        model_module = YnetLitModule(
            model,
            optimizer,
            lr_scheduler,
            train_loss,
            val_losses,
            test_losses,
            train_transform,
            val_transforms,
            test_transforms,
            normalized_clim,
        )
    elif architecture == "deepsd":
        elevation_list = prepare_deepsd_elevation(data_module, path_to_elevation)
        
        model_module = DeepSDLitModule(
            model,
            optimizer,
            lr_scheduler,
            train_loss,
            val_losses,
            test_losses,
            train_transform,
            val_transforms,
            test_transforms,
            elevation_list,
        )
    elif architecture == "dcgan":
        elevation = prepare_dcgan_elevation(data_module, path_to_elevation)
        model_module = GANLitModule(
            model,
            optimizer,
            lr_scheduler,
            train_loss,
            val_losses,
            test_losses,
            train_transform,
            val_transforms,
            test_transforms,
            elevation

        )
    elif architecture == "esrgan":
        model_module = ESRGANLitModule(
            model,
            optimizer,
            lr_scheduler,
            train_loss,
            val_losses,
            test_losses,
            train_transform,
            val_transforms,
            test_transforms,
        )
    else:
        model_module = LitModule(
            model,
            optimizer,
            lr_scheduler,
            train_loss,
            val_losses,
            test_losses,
            train_transform,
            val_transforms,
            test_transforms,
        )
    return model_module


load_forecasting_module = partial(
    load_model_module,
    task="forecasting",
    train_loss="lat_mse",
    val_loss=["lat_rmse", "lat_acc", "lat_mse"],
    test_loss=["lat_rmse", "lat_acc"],
    train_target_transform=None,
    val_target_transform=["denormalize", "denormalize", None],
    test_target_transform=["denormalize", "denormalize"],
)

load_climatebench_module = partial(
    load_model_module,
    task="forecasting",
    train_loss="mse",
    val_loss=["mse"],
    test_loss=["lat_nrmses", "lat_nrmseg", "lat_nrmse"],
    train_target_transform=None,
    val_target_transform=[nn.Identity()],
    test_target_transform=[nn.Identity(), nn.Identity(), nn.Identity()],
)

load_downscaling_module  = partial(
    load_model_module,
    task="downscaling",
    upsampling="bilinear",
    train_loss="mse",
    val_loss=["rmse", "pearson", "mean_bias", "mse", "PSNR", "SSIM", "KGE"],
    test_loss=["rmse", "pearson", "mean_bias", "PSNR", "SSIM", "KGE"],
    train_target_transform=None,
    val_target_transform=["denormalize", "denormalize", "denormalize", None, None, None, None],
    test_target_transform=["denormalize", "denormalize", "denormalize", None, None, None],
)


def load_architecture(task, data_module, architecture, upsampling,
                      optim_kwargs, model_kwargs, sched_kwargs):
    in_vars, out_vars = get_data_variables(data_module) 
    in_shape, out_shape = get_data_dims(data_module)
    
    def raise_not_impl():
        raise NotImplementedError(
            f"{architecture} is not an implemented architecture for the {task}"
            " task. If you think it should be, please raise an issue at"
            " https://github.com/aditya-grover/climate-learn/issues."
        )

    if task == "forecasting":
        history, in_channels, in_height, in_width = in_shape[1:]
        out_channels, out_height, out_width = out_shape[1:]
        if architecture.lower() == "climatology":
            norm = data_module.get_out_transforms()
            mean_norm = torch.tensor([norm[k].mean for k in norm.keys()])
            std_norm = torch.tensor([norm[k].std for k in norm.keys()])
            clim = get_climatology(data_module, "train")
            model = Climatology(clim, mean_norm, std_norm)
            optimizer = lr_scheduler = None
        elif architecture == "persistence":
            if not set(out_vars).issubset(in_vars):
                raise RuntimeError(
                    "Persistence requires the output variables to be a subset of"
                    " the input variables."
                )
            channels = [in_vars.index(o) for o in out_vars]
            model = Persistence(channels)
            optimizer = lr_scheduler = None
        elif architecture.lower() == "linear-regression":
            in_features = history * in_channels * in_height * in_width
            out_features = out_channels * out_height * out_width
            model = LinearRegression(in_features, out_features)
            optimizer = load_optimizer(model, "SGD", {"lr": 1e-5})
            lr_scheduler = None
        elif architecture.lower() == "rasp-theurey-2020":
            model = ResNet(
                in_channels=in_channels,
                out_channels=out_channels,
                history=history,
                hidden_channels=128,
                activation="leaky",
                norm=True,
                dropout=0.1,
                n_blocks=19,
            )
            optimizer = load_optimizer(
                model, "Adam", {"lr": 1e-5, "weight_decay": 1e-5}
            )
            lr_scheduler = None
        else:
            raise_not_impl()
    elif task == "downscaling":
        in_channels, in_height, in_width = in_shape[1:]
        out_channels, out_height, out_width = out_shape[1:]
        if architecture.lower() in (
            "bilinear-interpolation",
            "bicubic-interpolation",
            "nearest-interpolation",
        ):
            if len(out_vars) != len(in_vars):
                raise RuntimeError(
                    "Interpolation requires the output variables to match the"
                    " input variables."
                )
            interpolation_mode = architecture.split("-")[0]
            model = Interpolation((out_height, out_width), interpolation_mode)
            optimizer = lr_scheduler = None
        else:
            if architecture == "resnet":
                backbone = ResNet(in_channels, out_channels, n_blocks=28)
            elif architecture == "unet":
                backbone = Unet(
                    in_channels, out_channels,
                    ch_mults=[1, 2, 2],
                    n_blocks=2
                )
            elif architecture == "vit":
                backbone = VisionTransformer(
                    (out_height, out_width),
                    in_channels,
                    out_channels,
                    history=1,
                    patch_size=2,
                    learn_pos_emb=True,
                    embed_dim=128,
                    depth=8,
                    decoder_depth=2,
                    num_heads=4,
                    mlp_ratio=4,
                )
            elif architecture == "diffusion":
                backbone = GaussianDiffusion(
                    in_channels,
                    out_channels,
                    out_height,
                    out_width,
                    history=1,
                    timesteps=100,
                    # loss_type='l1',
                    beta_schedule='cosine',
                    res_rescale=out_width // in_width)
            elif architecture == "samvit":
                backbone = VisionTransformerSAM(
                    (out_height, out_width),
                    in_channels,
                    out_channels,
                    history=1,
                    patch_size=2,
                    embed_dim=128,
                    depth=4,
                    decoder_depth=1,
                    num_heads=4,
                    mlp_ratio=4,
                    neck_chans=0,
                )
            elif architecture == "ynet":
                backbone = YNet30(
                    in_channels,
                    out_channels,
                    num_layers=15,
                    num_features=64,
                    scale=out_width // in_width,
                )
            elif architecture == "deepsd":
                backbone = DeepSD(
                    in_channels,
                    out_channels,
                    num_features=64,
                    scale=out_width // in_width,
                )
            elif architecture == "dcgan":
                backbone = DCGAN(
                    in_channels,
                    out_channels,
                    scale=out_width // in_width,
                )
            elif architecture == "edrn":
                backbone = EDRN(
                    in_channels,
                    out_channels,
                    scale=out_width // in_width,
                )
            elif architecture == "esrgan":
                backbone = ESRGAN(
                    in_channels,
                    out_channels,
                    scale=out_width // in_width,
                )
            elif architecture == "hat":
                backbone = HAT(
                    in_channels,
                    out_channels,
                    scale=out_width // in_width,
                )
            else:
                raise_not_impl()
            if upsampling is None or upsampling.lower() == "none":
                model = backbone
            elif upsampling.lower() in ["bilinear", "bicubic", "nearest"]:
                model = nn.Sequential(
                    Interpolation((out_height, out_width), upsampling), backbone
                )
            elif upsampling.lower()[:4] == "unet":
                if "bilinear" in upsampling.lower().split("_"):
                    model = nn.Sequential(
                        UnetUpsampling((out_height, out_width), in_channels, bilinear=True), backbone
                    )
                else:
                    model = nn.Sequential(
                        UnetUpsampling((out_height, out_width), in_channels, bilinear=False), backbone
                    )
            else:
                raise_not_impl()
            
            if architecture == "dcgan" or architecture == "esrgan":
                optimizerG = load_optimizer(
                model.generator,
                "adam",
                optim_kwargs
                )
                optimizerD = load_optimizer(
                model.discriminator,
                "adam",
                # optim_kwargs
                {"lr": 2e-5, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
                )
                optimizer = (optimizerG, optimizerD)
                
                lr_schedulerG = load_lr_scheduler(
                                                "linear-warmup-cosine-annealing",
                                                optimizerG,
                                                sched_kwargs
                                                )
                lr_schedulerD = load_lr_scheduler(
                                                "linear-warmup-cosine-annealing",
                                                optimizerG,
                                                sched_kwargs
                                                )
                lr_scheduler = (lr_schedulerG, lr_schedulerD)
                
            else:
                optimizer = load_optimizer(
                model, "adamw",
                optim_kwargs
                # {"lr": 1e-5, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
                )

                lr_scheduler = load_lr_scheduler(
                    "linear-warmup-cosine-annealing",
                    optimizer,
                    sched_kwargs
                )
    return model, optimizer, lr_scheduler


def load_optimizer(net: torch.nn.Module, optim: str, optim_kwargs: Dict[str, Any] = {}):
    if len(list(net.parameters())) == 0:
        warnings.warn("Net has no trainable parameters, setting optimizer to `None`")
        optimizer = None
    if optim.lower() == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), **optim_kwargs)
    elif optim.lower() == "adam":
        optimizer = torch.optim.Adam(net.parameters(), **optim_kwargs)
    elif optim.lower() == "adamw":
        optimizer = torch.optim.AdamW(net.parameters(), **optim_kwargs)
    else:
        raise NotImplementedError(
            f"{optim} is not an implemented optimizer. If you think it should"
            " be, please raise an issue at"
            " https://github.com/aditya-grover/climate-learn/issues"
        )
    return optimizer


def load_lr_scheduler(
    sched: str, optimizer: torch.optim.Optimizer, sched_kwargs: Dict[str, Any] = {}
):
    if optimizer is None:
        warnings.warn("Optimizer is `None`, setting LR scheduler to `None` too")
        lr_scheduler = None
    if sched == "constant":
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, **sched_kwargs)
    elif sched == "linear":
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **sched_kwargs)
    elif sched == "exponential":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **sched_kwargs)
    elif sched == "linear-warmup-cosine-annealing":
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, **sched_kwargs)
    elif sched == "reduce-lr-on-plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **sched_kwargs
        )
    elif sched == "lambdaLR":
        lrdecay_lambda = lambda epoch: cosine_decay(epoch, **sched_kwargs)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lrdecay_lambda
        )
    else:
        raise NotImplementedError(
            f"{sched} is not an implemented learning rate scheduler. If you"
            " think it should be, please raise an issue at"
            " https://github.com/aditya-grover/climate-learn/issues"
        )
    return lr_scheduler


def load_loss(loss_name, aggregate_only, metainfo, **kwargs):
    loss_cls = METRICS_REGISTRY.get(loss_name, None)
    if loss_cls is None:
        raise NotImplementedError(
            f"{loss_name} is not an implemented loss. If you think it should be,"
            " please raise an issue at"
            " https://gtihub.com/aditya-grover/climate-learn/issues"
        )
    loss = loss_cls(aggregate_only=aggregate_only, metainfo=metainfo, **kwargs)
    return loss


def load_transform(transform_name, data_module):
    transform_cls = TRANSFORMS_REGISTRY.get(transform_name, None)
    if transform_cls is None:
        raise NotImplementedError(
            f"{transform_name} is not an implemented transform. If you think"
            " it should be, please raise an issue at"
            " https://github.com/aditya-grover/climate-learn/issues"
        )
    transform = transform_cls(data_module)
    return transform


def get_data_dims(data_module):
    return data_module.get_data_dims()


def get_gan_data_dims(data_module):
    return data_module.get_gan_data_dims()


def get_data_variables(data_module):
    return data_module.get_data_variables()


def get_climatology(data_module, split):
    clim = data_module.get_climatology(split=split)
    if clim is None:
        raise RuntimeError("Climatology has not yet been set.")
    # Hotfix to work with dict style data
    if isinstance(clim, dict):
        clim = torch.stack(tuple(clim.values()))
    return clim

def cosine_decay(epoch, warmup_epochs=100, max_epochs=10000):
    if epoch <= warmup_epochs:
        return (epoch / warmup_epochs)
    else:
        return 0.25 * (1 + np.cos((epoch - warmup_epochs)/max_epochs * np.pi))