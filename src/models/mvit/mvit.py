# --------------------------------------------------------
# the model file has been built on top of the following:
#   https://github.com/facebookresearch/SlowFast/blob/main/slowfast/models/video_model_builder.py
# --------------------------------------------------------


import numpy as np
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from .utils import (
    MultiScaleBlock,
    PatchEmbed,
    calc_mvit_feature_geometry,
    get_3d_sincos_pos_embed,
    round_width,
    )

from ..build import MODEL_REGISTRY

def get_mvit_config(config, mtype):
    cfg = config.clone()

    if mtype == 'audio':
        temporal_size = 1
        in_chans = 1
        spatial_size = cfg.AUDIO.CROP_SIZE

        cfg.MVIT.PATCH_2D = True
        cfg.MVIT.ZERO_DECAY_POS_CLS = False
        cfg.MVIT.MODE = "conv"
        cfg.MVIT.CLS_EMBED_ON = False

        if cfg.MVIT.PATCH_OVERLAP:
            cfg.MVIT.PATCH_KERNEL = [7, 7]
            cfg.MVIT.PATCH_STRIDE = [4, 4]
            cfg.MVIT.PATCH_PADDING = [3, 3]
        else:
            cfg.MVIT.PATCH_KERNEL = [7, 9]
            cfg.MVIT.PATCH_STRIDE = [4, 8]
            cfg.MVIT.PATCH_PADDING = [3, 4]

        cfg.MVIT.EMBED_DIM = 96
        cfg.MVIT.NUM_HEADS = 1
        cfg.MVIT.MLP_RATIO = 4.0
        cfg.MVIT.QKV_BIAS = True
        cfg.MVIT.NORM = "layernorm"
        cfg.MVIT.POOL_KVQ_KERNEL = [1, 3, 3]

        if cfg.MVIT.SIZE == "S":
            # https://github.com/facebookresearch/SlowFast/blob/main/configs/ImageNet/MVITv2_S.yaml
            cfg.MVIT.DROPPATH_RATE = 0.1
            cfg.MVIT.DEPTH = 16
            cfg.MVIT.DIM_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]
            cfg.MVIT.HEAD_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]
            cfg.MVIT.POOL_KV_STRIDE = [[0, 1, 4, 4], [1, 1, 2, 2], [2, 1, 2, 2], [3, 1, 1, 1], [4, 1, 1, 1], [5, 1, 1, 1], [6, 1, 1, 1], [7, 1, 1, 1], [8, 1, 1, 1], [9, 1, 1, 1], [10, 1, 1, 1], [11, 1, 1, 1], [12, 1, 1, 1], [13, 1, 1, 1], [14, 1, 1, 1], [15, 1, 1, 1]]
            cfg.MVIT.POOL_Q_STRIDE = [[0, 1, 1, 1], [1, 1, 2, 2], [2, 1, 1, 1], [3, 1, 2, 2], [4, 1, 1, 1], [5, 1, 1, 1], [6, 1, 1, 1], [7, 1, 1, 1], [8, 1, 1, 1], [9, 1, 1, 1], [10, 1, 1, 1], [11, 1, 1, 1], [12, 1, 1, 1], [13, 1, 1, 1], [14, 1, 2, 2], [15, 1, 1, 1]]

        elif cfg.MVIT.SIZE == "B":
            # https://github.com/facebookresearch/SlowFast/blob/main/configs/ImageNet/MVITv2_B.yaml
            cfg.MVIT.DROPPATH_RATE = 0.3
            cfg.MVIT.DEPTH = 24
            cfg.MVIT.DIM_MUL = [[2, 2.0], [5, 2.0], [21, 2.0]]
            cfg.MVIT.HEAD_MUL = [[2, 2.0], [5, 2.0], [21, 2.0]]
            cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE = [1, 4, 4]
            cfg.MVIT.POOL_Q_STRIDE = [[0, 1, 1, 1], [1, 1, 1, 1], [2, 1, 2, 2], [3, 1, 1, 1], [4, 1, 1, 1], [5, 1, 2, 2], [6, 1, 1, 1], [7, 1, 1, 1], [8, 1, 1, 1], [9, 1, 1, 1], [10, 1, 1, 1], [11, 1, 1, 1], [12, 1, 1, 1], [13, 1, 1, 1], [14, 1, 1, 1], [15, 1, 1, 1], [16, 1, 1, 1], [17, 1, 1, 1], [18, 1, 1, 1], [19, 1, 1, 1], [20, 1, 1, 1], [21, 1, 2, 2], [22, 1, 1, 1], [23, 1, 1, 1]]
        else:
            raise NotImplementedError

        cfg.MVIT.RESIDUAL_POOLING = True
        cfg.MVIT.USE_ABS_POS = False
        cfg.MVIT.REL_POS_SPATIAL = True
        cfg.MVIT.DIM_MUL_IN_ATT = True

    elif mtype == 'video':
        temporal_size = cfg.VIDEO.N_FRAMES
        in_chans = 3
        spatial_size = cfg.VIDEO.CROP_SIZE

        cfg.MVIT.ZERO_DECAY_POS_CLS = False
        cfg.MVIT.USE_ABS_POS = False
        cfg.MVIT.REL_POS_SPATIAL = True
        cfg.MVIT.REL_POS_TEMPORAL = True
        cfg.MVIT.NUM_HEADS = 1
        cfg.MVIT.EMBED_DIM = 96

        if cfg.MVIT.PATCH_OVERLAP:
            cfg.MVIT.PATCH_KERNEL = (3, 7, 7)
            cfg.MVIT.PATCH_STRIDE = (2, 4, 4)
            cfg.MVIT.PATCH_PADDING = (1, 3, 3)
        else:
            cfg.MVIT.PATCH_KERNEL = (3, 9, 9)
            cfg.MVIT.PATCH_STRIDE = (2, 8, 8)
            cfg.MVIT.PATCH_PADDING = (1, 4, 4)

        cfg.MVIT.MLP_RATIO = 4.0
        cfg.MVIT.QKV_BIAS = True
        cfg.MVIT.NORM = "layernorm"
        cfg.MVIT.MODE = "conv"
        cfg.MVIT.CLS_EMBED_ON = True

        if cfg.MVIT.SIZE == "S":
            # https://github.com/facebookresearch/SlowFast/blob/main/configs/Kinetics/MVITv2_S_16x4.yaml
            cfg.MVIT.DEPTH = 16
            cfg.MVIT.DROPPATH_RATE = 0.2
            cfg.MVIT.DIM_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]
            cfg.MVIT.HEAD_MUL = [[1, 2.0], [3, 2.0], [14, 2.0]]
            cfg.MVIT.POOL_Q_STRIDE = [[0, 1, 1, 1], [1, 1, 2, 2], [2, 1, 1, 1], [3, 1, 2, 2], [4, 1, 1, 1], [5, 1, 1, 1], [6, 1, 1, 1], [7, 1, 1, 1], [8, 1, 1, 1], [9, 1, 1, 1], [10, 1, 1, 1], [11, 1, 1, 1], [12, 1, 1, 1], [13, 1, 1, 1], [14, 1, 2, 2], [15, 1, 1, 1]]

        elif cfg.MVIT.SIZE == "B":
            # https://github.com/facebookresearch/SlowFast/blob/main/configs/Kinetics/MVITv2_B_32x3.yaml
            cfg.MVIT.DEPTH = 24
            cfg.MVIT.DROPPATH_RATE = 0.3
            cfg.MVIT.DIM_MUL = [[2, 2.0], [5, 2.0], [21, 2.0]]
            cfg.MVIT.HEAD_MUL = [[2, 2.0], [5, 2.0], [21, 2.0]]
            cfg.MVIT.POOL_Q_STRIDE = [[0, 1, 1, 1], [1, 1, 1, 1], [2, 1, 2, 2], [3, 1, 1, 1], [4, 1, 1, 1], [5, 1, 2, 2], [6, 1, 1, 1], [7, 1, 1, 1], [8, 1, 1, 1], [9, 1, 1, 1], [10, 1, 1, 1], [11, 1, 1, 1], [12, 1, 1, 1], [13, 1, 1, 1], [14, 1, 1, 1], [15, 1, 1, 1], [16, 1, 1, 1], [17, 1, 1, 1], [18, 1, 1, 1], [19, 1, 1, 1], [20, 1, 1, 1], [21, 1, 2, 2], [22, 1, 1, 1], [23, 1, 1, 1]]

        else:
            raise NotImplementedError

        cfg.MVIT.POOL_KVQ_KERNEL = [3, 3, 3]
        cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE = [1, 8, 8]
        cfg.MVIT.DROPOUT_RATE = 0.0
        cfg.MVIT.DIM_MUL_IN_ATT = True
        cfg.MVIT.RESIDUAL_POOLING = True

    else:
        raise NotImplementedError
    return cfg, (temporal_size, spatial_size, in_chans)



class MultiScaleViT(nn.Module):

    def __init__(self, cfg, temporal_size, spatial_size, in_chans):
        super().__init__()
        # Get parameters.

        self.cfg = cfg
        pool_first = cfg.MVIT.POOL_FIRST
        # Prepare input.
        self.use_2d_patch = cfg.MVIT.PATCH_2D
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if self.use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        self.T = temporal_size // self.patch_stride[0]
        self.H = spatial_size[0] // self.patch_stride[1]
        self.W = spatial_size[1] // self.patch_stride[2]
        # Prepare output.
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        layer_scale_init_value = cfg.MVIT.LAYER_SCALE_INIT_VALUE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.use_mean_pooling = cfg.MVIT.USE_MEAN_POOLING
        # Params for positional embedding
        self.use_abs_pos = cfg.MVIT.USE_ABS_POS
        self.use_fixed_sincos_pos = cfg.MVIT.USE_FIXED_SINCOS_POS
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        self.rel_pos_spatial = cfg.MVIT.REL_POS_SPATIAL
        self.rel_pos_temporal = cfg.MVIT.REL_POS_TEMPORAL
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.patch_embed = PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=self.use_2d_patch,
        )

        self.input_dims = [temporal_size, *spatial_size]
        # assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = np.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            if self.sep_pos_embed:
                self.pos_embed_spatial = nn.Parameter(
                    torch.zeros(
                        1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                    )
                )
                self.pos_embed_temporal = nn.Parameter(
                    torch.zeros(1, self.patch_dims[0], embed_dim)
                )
                if self.cls_embed_on:
                    self.pos_embed_class = nn.Parameter(
                        torch.zeros(1, 1, embed_dim)
                    )
            else:
                self.pos_embed = nn.Parameter(
                    torch.zeros(
                        1,
                        pos_embed_dim,
                        embed_dim,
                    ),
                    requires_grad=not self.use_fixed_sincos_pos,
                )

        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
        for i in range(len(cfg.MVIT.DIM_MUL)):
            dim_mul[cfg.MVIT.DIM_MUL[i][0]] = cfg.MVIT.DIM_MUL[i][1]
        for i in range(len(cfg.MVIT.HEAD_MUL)):
            head_mul[cfg.MVIT.HEAD_MUL[i][0]] = cfg.MVIT.HEAD_MUL[i][1]

        pool_q = [[] for i in range(cfg.MVIT.DEPTH)]
        pool_kv = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_q = [[] for i in range(cfg.MVIT.DEPTH)]
        stride_kv = [[] for i in range(cfg.MVIT.DEPTH)]

        for i in range(len(cfg.MVIT.POOL_Q_STRIDE)):
            stride_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_Q_STRIDE[i][
                1:
            ]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_q[cfg.MVIT.POOL_Q_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s for s in cfg.MVIT.POOL_Q_STRIDE[i][1:]
                ]

        # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
        if cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE is not None:
            _stride_kv = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE
            cfg.MVIT.POOL_KV_STRIDE = []
            for i in range(cfg.MVIT.DEPTH):
                if len(stride_q[i]) > 0:
                    _stride_kv = [
                        max(_stride_kv[d] // stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                cfg.MVIT.POOL_KV_STRIDE.append([i] + _stride_kv)

        for i in range(len(cfg.MVIT.POOL_KV_STRIDE)):
            stride_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = cfg.MVIT.POOL_KV_STRIDE[
                i
            ][1:]
            if cfg.MVIT.POOL_KVQ_KERNEL is not None:
                pool_kv[
                    cfg.MVIT.POOL_KV_STRIDE[i][0]
                ] = cfg.MVIT.POOL_KVQ_KERNEL
            else:
                pool_kv[cfg.MVIT.POOL_KV_STRIDE[i][0]] = [
                    s + 1 if s > 1 else s
                    for s in cfg.MVIT.POOL_KV_STRIDE[i][1:]
                ]

        self.pool_q = pool_q
        self.pool_kv = pool_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        input_size = self.patch_dims

        self.blocks = nn.ModuleList()

        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            if cfg.MVIT.DIM_MUL_IN_ATT:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i],
                    divisor=round_width(num_heads, head_mul[i]),
                )
            else:
                dim_out = round_width(
                    embed_dim,
                    dim_mul[i + 1],
                    divisor=round_width(num_heads, head_mul[i + 1]),
                )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_rate=self.drop_rate,
                drop_path=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode=mode,
                has_cls_embed=self.cls_embed_on,
                pool_first=pool_first,
                rel_pos_spatial=self.rel_pos_spatial,
                rel_pos_temporal=self.rel_pos_temporal,
                rel_pos_zero_init=cfg.MVIT.REL_POS_ZERO_INIT,
                residual_pooling=cfg.MVIT.RESIDUAL_POOLING,
                dim_mul_in_att=cfg.MVIT.DIM_MUL_IN_ATT,
                separate_qkv=cfg.MVIT.SEPARATE_QKV,
            )

            self.blocks.append(attention_block)
            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride
                    for size, stride in zip(input_size, stride_q[i])
                ]

            embed_dim = dim_out

        self.norm = norm_layer(embed_dim)
        self.OUT_DIM = embed_dim

        if self.use_abs_pos:
            if self.sep_pos_embed:
                trunc_normal_(self.pos_embed_spatial, std=0.02)
                trunc_normal_(self.pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.pos_embed, std=0.02)
                if self.use_fixed_sincos_pos:
                    pos_embed = get_3d_sincos_pos_embed(
                        self.pos_embed.shape[-1],
                        self.H,
                        self.T,
                        cls_token=self.cls_embed_on,
                    )
                    self.pos_embed.data.copy_(
                        torch.from_numpy(pos_embed).float().unsqueeze(0)
                    )

        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        self.feat_size, self.feat_stride = calc_mvit_feature_geometry(cfg, temporal_size, spatial_size)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.02)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.use_abs_pos:
                if self.sep_pos_embed:
                    names.extend(
                        [
                            "pos_embed_spatial",
                            "pos_embed_temporal",
                            "pos_embed_class",
                        ]
                    )
                else:
                    names.append("pos_embed")
            if self.rel_pos_spatial:
                names.extend(["rel_pos_h", "rel_pos_w", "rel_pos_hw"])
            if self.rel_pos_temporal:
                names.extend(["rel_pos_t"])
            if self.cls_embed_on:
                names.append("cls_token")

        return names

    def _get_pos_embed(self, pos_embed, bcthw):

        if len(bcthw) == 4:
            t, h, w = 1, bcthw[-2], bcthw[-1]
        else:
            t, h, w = bcthw[-3], bcthw[-2], bcthw[-1]
        if self.cls_embed_on:
            cls_pos_embed = pos_embed[:, 0:1, :]
            pos_embed = pos_embed[:, 1:]
        txy_num = pos_embed.shape[1]
        p_t, p_h, p_w = self.patch_dims
        assert p_t * p_h * p_w == txy_num

        if (p_t, p_h, p_w) != (t, h, w):
            new_pos_embed = F.interpolate(
                pos_embed[:, :, :]
                .reshape(1, p_t, p_h, p_w, -1)
                .permute(0, 4, 1, 2, 3),
                size=(t, h, w),
                mode="trilinear",
            )
            pos_embed = new_pos_embed.reshape(1, -1, t * h * w).permute(0, 2, 1)

        if self.cls_embed_on:
            pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)

        return pos_embed

    @torch.inference_mode()
    def inference(self, x):
        # this is used for feature extraction
        assert isinstance(x, (torch.Tensor, list))
        if isinstance(x, list):
            # if we use multi-crop for video, this will be a list
            y = list(map(self.forward, x))
            y = torch.stack(y, dim=1)
        else:
            y = self.forward(x)
        return y

    def forward(self, x, inference=False):
        if inference:
            return self.inference(x)
            
        x, bcthw = self.patch_embed(x)
        bcthw = list(bcthw)
        if len(bcthw) == 4:  # Fix bcthw in case of 4D tensor
            bcthw.insert(2, torch.tensor(self.T))
        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        assert len(bcthw) == 5 and (T, H, W) == (self.T, self.H, self.W), bcthw
        B, N, C = x.shape

        s = 1 if self.cls_embed_on else 0
        if self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # s: on/off cls token

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            if self.use_fixed_sincos_pos:
                cls_tokens = cls_tokens + self.pos_embed[:, :s, :]
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x += self._get_pos_embed(pos_embed, bcthw)
            else:
                x += self._get_pos_embed(self.pos_embed, bcthw)

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]

        for blk in self.blocks:
            x, thw = blk(x, thw)

        if self.use_mean_pooling:
            if self.cls_embed_on:
                x = x[:, 1:]
            x = x.mean(1)
            x = self.norm(x)
        elif self.cls_embed_on:
            x = self.norm(x)
            x = x[:, 0]
        else:  # this is default, [norm->mean]
            x = self.norm(x)
            x = x.mean(1)

        return x



@MODEL_REGISTRY.register()
class MViT(MultiScaleViT):
    def __init__(self, cfg, mtype):
        cfg, args = get_mvit_config(cfg, mtype)
        super(MViT, self).__init__(cfg, *args)
