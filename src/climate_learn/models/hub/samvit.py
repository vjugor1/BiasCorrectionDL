# Standard library
from functools import partial
from typing import Callable, Optional, Tuple

# Local application
from .components.samvit_blocks import Block, checkpoint_seq, PatchEmbed
from .utils import register

# Third party
import torch
import torch.nn as nn
import torch.utils.checkpoint
from timm.layers import (
    Mlp,
    Format,
    PatchDropout,
    LayerNorm2d,
    RotaryEmbeddingCat,
    to_2tuple,
)


@register("samvit")
class VisionTransformerSAM(nn.Module):
    """Vision Transformer for Segment-Anything Model(SAM)

    A PyTorch impl of : `Exploring Plain Vision Transformer Backbones for Object Detection` or `Segment Anything Model (SAM)`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        history,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        decoder_depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        pre_norm: bool = False,
        pos_drop_rate: float = 0.1,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        embed_layer: Callable = partial(
            PatchEmbed, output_fmt=Format.NHWC, strict_img_size=False
        ),
        norm_layer: Optional[Callable] = nn.LayerNorm,
        act_layer: Optional[Callable] = nn.GELU,
        block_fn: Callable = Block,
        mlp_layer: Callable = Mlp,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        use_rope: bool = False,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (),
        neck_chans: int = 256,
        ref_feat_shape: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels * history
        self.out_channels = out_channels
        self.patch_size = patch_size
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used
        )
        grid_size = self.patch_embed.grid_size

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, grid_size[0], grid_size[1], embed_dim, requires_grad=True
                )
            )
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=0,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        if use_rope:
            assert (
                not use_rel_pos
            ), "ROPE and relative pos embeddings should not be enabled at same time"
            if ref_feat_shape is not None:
                assert len(ref_feat_shape) == 2
                ref_feat_shape_global = to_2tuple(ref_feat_shape[0])
                ref_feat_shape_window = to_2tuple(ref_feat_shape[1])
            else:
                ref_feat_shape_global = ref_feat_shape_window = None
            self.rope_global = RotaryEmbeddingCat(
                embed_dim // num_heads,
                in_pixels=False,
                feat_shape=grid_size,
                ref_feat_shape=ref_feat_shape_global,
            )
            self.rope_window = RotaryEmbeddingCat(
                embed_dim // num_heads,
                in_pixels=False,
                feat_shape=to_2tuple(window_size),
                ref_feat_shape=ref_feat_shape_window,
            )
        else:
            self.rope_global = None
            self.rope_window = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    use_rel_pos=use_rel_pos,
                    window_size=window_size if i not in global_attn_indexes else 0,
                    input_size=grid_size,
                    rope=(
                        self.rope_window
                        if i not in global_attn_indexes
                        else self.rope_global
                    ),
                )
                for i in range(depth)
            ]
        )
        if neck_chans:
            self.neck = nn.Sequential(
                nn.Conv2d(
                    embed_dim,
                    neck_chans,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(neck_chans),
                nn.Conv2d(
                    neck_chans,
                    neck_chans,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm2d(neck_chans),
            )
            self.num_features = neck_chans
        else:
            self.neck = LayerNorm2d(embed_dim)
            neck_chans = embed_dim

        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, out_channels * patch_size**2))
        self.head = nn.Sequential(*self.head)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "dist_token"}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def unpatchify(self, x: torch.Tensor):
        """
        x: (B, Hp, Wp, V * patch_size**2)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        c = self.out_channels
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        assert h * w == x.shape[1] * x.shape[2]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_encoder(self, x: torch.Tensor):
        # x.shape = [B,C,H,W]
        x = self.patch_embed(x)
        # x.shape = [B,H // p, W // p,embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.neck(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # x.shape = [B, H // p, W // p, neck_dim]
        return x

    def forward(self, x):
        if len(x.shape) == 5:  # x.shape = [B,T,in_channels,H,W]
            x = x.flatten(1, 2)
        # x.shape = [B,T*in_channels,H,W]
        x = self.forward_encoder(x)
        # x.shape = [B, H // p, W // p, neck_dim]
        x = self.head(x)
        # x.shape = [B, H // p, W // p, V * patch_size**2]
        preds = self.unpatchify(x)
        # preds.shape = [B,out_channels,H,W]
        return preds
