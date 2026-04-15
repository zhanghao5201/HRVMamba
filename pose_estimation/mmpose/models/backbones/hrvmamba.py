import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from mmengine.model import BaseModule
from mmengine.model.weight_init import trunc_normal_
from mmengine.runner import load_state_dict
from mmengine.utils import to_2tuple
from mmpose.registry import MODELS
from mmpose.utils import get_root_logger
from .utils import get_state_dict
import pdb
import DCNv4

from mmpose.models.backbones.utils import channel_shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from torchvision.models import VisionTransformer

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


import pdb
try:
    from .csm_triton_new import cross_scan_fn, cross_merge_fn
except:
    from csm_triton_new import cross_scan_fn, cross_merge_fn

try:
    from .csms6s_new import selective_scan_fn, selective_scan_flop_jit
except:
    from csms6s_new import selective_scan_fn, selective_scan_flop_jit

try:
    from .mamba2.ssd_minimal import selective_scan_chunk_fn
except:
    from mamba2.ssd_minimal import selective_scan_chunk_fn
    
from .hrnet import Bottleneck,HRModule, HRNet
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer

class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x

class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError
               
class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True) 
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  #
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):#
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0)) 
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0)) # (K, inner)
        del dt_projs

        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True) 
        Ds = cls.D_init(d_inner, copies=k_group, merge=True) # (K * D)  
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


class SS2Dv2:
    def __initv2__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3, 
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        forward_type="v2",
        channel_first=False,
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)#256
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank 
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2
        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)#
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, d_inner, channel_first)
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba", scan_force_torch=True),
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            v04=partial(self.forward_corev2, force_fp32=False), # 
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),  
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 4

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        core_op=getattr(DCNv4, 'DCNv4')
        if d_model%80==0:
            dw_kernel_size=3
            offset_scale=1.0
        else:
            dw_kernel_size=None
            offset_scale=1.0
    
        
        self.conv2d=core_op(
            channels=d_inner,
            group=dt_rank,
            offset_scale=offset_scale,
            dw_kernel_size=dw_kernel_size, 
            output_bias=False#
        )

        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) #
        del self.x_proj        
        
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4,
            )
        elif initialize in ["v1"]:
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((k_group, d_inner, dt_rank))) # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((k_group, d_inner))) # 0.1 is added in 0430
        elif initialize in ["v2"]:
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))

    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        force_fp32=False, 
        ssoflex=True, 
        no_einsum=False,
        selective_scan_backend = None,
        scan_mode = "cross2d",
        scan_force_torch = False,
        **kwargs,
    ):
        assert scan_mode in ["unidi", "bidi", "cross2d", "cascade2d"]
        assert selective_scan_backend in [None, "oflex", "core", "mamba", "torch"]
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape #
        D, N = self.A_logs.shape #torch.Size([1024, 1])
        K, D, R = self.dt_projs_weight.shape #torch.Size([4, 256, 8])
        L = H * W #14*14
        
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex, backend=selective_scan_backend)
        
        if _scan_mode == 3:
            x_proj_bias = getattr(self, "x_proj_bias", None)
            def scan_rowcol(
                x: torch.Tensor, 
                proj_weight: torch.Tensor, 
                proj_bias: torch.Tensor, 
                dt_weight: torch.Tensor, 
                dt_bias: torch.Tensor, # 
                _As: torch.Tensor, # 
                _Ds: torch.Tensor,
                width = True,
            ):
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2) # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1), bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys
            
            As = -self.A_logs.to(torch.float).exp().view(4, -1, N)
            x = F.layer_norm(x.permute(0, 2, 3, 1), normalized_shape=(int(x.shape[1]),)).permute(0, 3, 1, 2).contiguous() # added0510 to avoid nan
            y_row = scan_rowcol(
                x,
                proj_weight = self.x_proj_weight.view(4, -1, D)[:2].contiguous(), 
                proj_bias = (x_proj_bias.view(4, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight = self.dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                dt_bias = (self.dt_projs_bias.view(4, -1)[:2].contiguous() if self.dt_projs_bias is not None else None),
                _As = As[:2].contiguous().view(-1, N),
                _Ds = self.Ds.view(4, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3) # (B,C,H,W)
            y_row = F.layer_norm(y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)).permute(0, 3, 1, 2).contiguous() # added0510 to avoid nan
            y_col = scan_rowcol(
                y_row,
                proj_weight = self.x_proj_weight.view(4, -1, D)[2:].contiguous().to(y_row.dtype), 
                proj_bias = (x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if x_proj_bias is not None else None),
                dt_weight = self.dt_projs_weight.view(4, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias = (self.dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype) if self.dt_projs_bias is not None else None),
                _As = As[2:].contiguous().view(-1, N),
                _Ds = self.Ds.view(4, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1)
            y = y_col
        else:
            x_proj_bias = getattr(self, "x_proj_bias", None)
            xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L), self.x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                dts = F.conv1d(dts.contiguous().view(B, -1, L), self.dt_projs_weight.view(K * D, -1, 1), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)
            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L) 
            As = -self.A_logs.to(torch.float).exp() 
            Ds = self.Ds.to(torch.float) # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            )
            ys=ys.view(B, K, -1, H, W)
            
            y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode, force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1) 
        y = out_norm(y)

        return y.to(x.dtype)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) #
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()

        x=x.permute(0,2,3,1)
        x = self.conv2d(x) # 
        x=x.permute(0, 3, 1, 2)

        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value

class SS2D(nn.Module, SS2Dv2):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        super().__init__()
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type), **kwargs)
        elif forward_type.startswith("xv"):
            self.__initxv__(**kwargs)
        elif forward_type.startswith("m"):
            self.__initm0__(**kwargs)
        else:
            self.__initv2__(**kwargs)

def _split_channels(channels, num_groups):
    split_channels = [channels // num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels

class HRVmambaBlock(BaseModule):
    expansion = 1
    def __init__(
        self,
        hidden_dim: int = 0,
        out_features: int = 0,
        drop_path: float = 0,
        norm_layer = LayerNorm2d,
        channel_first=True,
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        use_checkpoint: bool = False,
        post_norm: bool = False,
        init_cfg=None,
        **kwargs,
    ):
        super(HRVmambaBlock,self).__init__(init_cfg=init_cfg)
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        self.num_groups=4#1124
        self.split_channels = _split_channels(hidden_dim, self.num_groups)
        self.conv2d = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.split_channels[i],
                out_channels=self.split_channels[i],
                kernel_size=i * 2 + 3,
                padding=i + 1,
                groups=self.split_channels[i],
                )
            for i in range(self.num_groups)
        ])
        self.norm0 = norm_layer(hidden_dim)
        self.act0=nn.GELU()
        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim, 
                d_state=ssm_d_state, #1
                ssm_ratio=ssm_ratio, #2.0
                dt_rank=ssm_dt_rank, #16
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv, #3
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.post_norm:
            x_split = torch.split(x, self.split_channels, dim=1)
            x = [conv(t) for conv, t in zip(self.conv2d, x_split)]
            x = torch.cat(x, dim=1)
            x = channel_shuffle(x, self.num_groups)
            x=input+self.act0(self.norm0(x))
        else:
            x=self.norm0(x)
            x_split = torch.split(x, self.split_channels, dim=1)
            x = [conv(t) for conv, t in zip(self.conv2d, x_split)]
            x = torch.cat(x, dim=1)
            x = channel_shuffle(x, self.num_groups)
            x=self.act0(x)+input

        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)



class HRVmambaModule(HRModule):
    def __init__(self,
                 num_branches,
                 block,
                 num_blocks,
                 num_inchannels,
                 num_channels,
                 multiscale_output,
                 drop_paths, 
                 ssm_d_state,
                 ssm_ratio,
                 ssm_dt_rank,
                 ssm_act_layer,        
                 ssm_conv,
                 ssm_conv_bias,
                 ssm_drop_rate, 
                 ssm_init,
                 forward_type,
                 mlp_ratio,
                 mlp_act_layer,
                 mlp_drop_rate,
                 gmlp,                           
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),                 
                 upsample_cfg=dict(mode='bilinear', align_corners=False),
                 **kwargs):
        
        self.drop_paths = drop_paths
        self.ssm_d_state=ssm_d_state
        self.ssm_ratio= ssm_ratio
        self.ssm_dt_rank=ssm_dt_rank
        self.ssm_act_layer=ssm_act_layer        
        self.ssm_conv=ssm_conv
        self.ssm_conv_bias=ssm_conv_bias
        self.ssm_drop_rate=ssm_drop_rate 
        self.ssm_init=ssm_init
        self.forward_type=forward_type
        self.mlp_ratio=mlp_ratio
        self.mlp_act_layer=mlp_act_layer
        self.mlp_drop_rate=mlp_drop_rate
        self.gmlp=gmlp

        super().__init__(num_branches, block, num_blocks, num_inchannels,
                         num_channels, multiscale_output,  conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg, upsample_cfg=upsample_cfg, **kwargs)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        """Build one branch."""
        assert stride == 1 and self.in_channels[branch_index] == num_channels[
            branch_index]
        layers = []
        layers.append(
            block(
                self.in_channels[branch_index],
                num_channels[branch_index],
                ssm_d_state=self.ssm_d_state,
                ssm_ratio=self.ssm_ratio,
                ssm_dt_rank=self.ssm_dt_rank,
                ssm_act_layer=self.ssm_act_layer,        
                ssm_conv=self.ssm_conv,
                ssm_conv_bias=self.ssm_conv_bias,
                ssm_drop_rate=self.ssm_drop_rate, 
                ssm_init=self.ssm_init,
                forward_type=self.forward_type,
                # =========================
                mlp_ratio=self.mlp_ratio,
                mlp_act_layer=self.mlp_act_layer,
                mlp_drop_rate=self.mlp_drop_rate,
                gmlp=self.gmlp,
                drop_path=self.drop_paths[0],
                norm_cfg=self.norm_cfg,
                ))

        self.in_channels[
            branch_index] = self.in_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index],
                    ssm_d_state=self.ssm_d_state,
                    ssm_ratio=self.ssm_ratio,
                    ssm_dt_rank=self.ssm_dt_rank,
                    ssm_act_layer=self.ssm_act_layer,        
                    ssm_conv=self.ssm_conv,
                    ssm_conv_bias=self.ssm_conv_bias,
                    ssm_drop_rate=self.ssm_drop_rate, 
                    ssm_init=self.ssm_init,
                    forward_type=self.forward_type,
                    # =========================
                    mlp_ratio=self.mlp_ratio,
                    mlp_act_layer=self.mlp_act_layer,
                    mlp_drop_rate=self.mlp_drop_rate,
                    gmlp=self.gmlp,
                    # =========================      

                    drop_path=self.drop_paths[i],
                    norm_cfg=self.norm_cfg,
                    ))
        return nn.Sequential(*layers)

    def _make_fuse_layers(self):
        """Build fuse layers."""
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.in_channels
        fuse_layers = []
        for i in range(num_branches if self.multiscale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_inchannels[j],
                                num_inchannels[i],
                                kernel_size=1,
                                stride=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_inchannels[i])[1],
                            nn.Upsample(
                                scale_factor=2**(j - i),
                                mode=self.upsample_cfg['mode'],
                                align_corners=self.
                                upsample_cfg['align_corners'])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            with_out_act = False
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            with_out_act = True
                        sub_modules = [
                            build_conv_layer(
                                self.conv_cfg,
                                num_inchannels[j],
                                num_inchannels[j],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                groups=num_inchannels[j],
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg,
                                             num_inchannels[j])[1],
                            build_conv_layer(
                                self.conv_cfg,
                                num_inchannels[j],
                                num_outchannels_conv3x3,
                                kernel_size=1,
                                stride=1,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg,
                                             num_outchannels_conv3x3)[1]
                        ]
                        if with_out_act:
                            sub_modules.append(nn.ReLU(False))
                        conv3x3s.append(nn.Sequential(*sub_modules))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        """Return the number of input channels."""
        return self.in_channels
    

@MODELS.register_module()
class HRVmamba(HRNet):
    blocks_dict = {'BOTTLENECK': Bottleneck,'HRVmambaBlock': HRVmambaBlock}
    act_layer_dict={"silu": nn.SiLU,"gelu": nn.GELU }
    def __init__(
        self,
        extra,
        in_channels=3,
        conv_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=False,
        zero_init_residual=False,
        frozen_stages=-1,
        init_cfg=[
            dict(type='Normal', std=0.001, layer=['Conv2d']),
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
        ],
    ):
        depths = [
            extra[stage]['num_blocks'][0] * extra[stage]['num_modules']
            for stage in ['stage2', 'stage3', 'stage4']
        ]
        depth_s2, depth_s3, _ = depths
        drop_path_rate = extra['drop_path_rate']
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        extra['stage2']['drop_path_rates'] = dpr[0:depth_s2]
        extra['stage3']['drop_path_rates'] = dpr[depth_s2:depth_s2 + depth_s3]
        extra['stage4']['drop_path_rates'] = dpr[depth_s2 + depth_s3:]
        upsample_cfg = extra.get('upsample', {
            'mode': 'bilinear',
            'align_corners': False
        })
        extra['upsample'] = upsample_cfg

        super().__init__(extra, in_channels, conv_cfg, norm_cfg, norm_eval,
                         with_cp, zero_init_residual, frozen_stages, init_cfg)
    def _make_stage(self,
                    layer_config,
                    num_inchannels,
                    multiscale_output=True):
        """Make each stage."""
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        ssm_d_state=layer_config['ssm_d_state']
        ssm_ratio= layer_config['ssm_ratio']
        ssm_dt_rank=layer_config['ssm_dt_rank']
        ssm_act_layer=self.act_layer_dict[layer_config['ssm_act_layer'] ]      
        ssm_conv=layer_config['ssm_conv']
        ssm_conv_bias=layer_config['ssm_conv_bias']
        ssm_drop_rate=layer_config['ssm_drop_rate'] 
        ssm_init=layer_config['ssm_init']
        forward_type=layer_config['forward_type']
        mlp_ratio=layer_config['mlp_ratio']
        mlp_act_layer=self.act_layer_dict[layer_config['mlp_act_layer']]
        mlp_drop_rate=layer_config['mlp_drop_rate']
        gmlp=layer_config['gmlp']        
        
        drop_path_rates = layer_config['drop_path_rates']

        modules = []
        for i in range(num_modules):
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True
            modules.append(
                HRVmambaModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,                    
                    reset_multiscale_output,
                    drop_paths=drop_path_rates[num_blocks[0] *
                                               i:num_blocks[0] * (i + 1)],
                    # =========================
                    ssm_d_state=ssm_d_state,
                    ssm_ratio= ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer,        
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate, 
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    # =========================
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                    # =========================
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,         
                    upsample_cfg=self.upsample_cfg))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels
