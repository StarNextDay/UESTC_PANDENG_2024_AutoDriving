"""
author  :   lifangbo
date    :   2024-04-25
"""
from typing import Callable, Optional

import torch
from torch import Tensor
from torch import nn


class Mlp(nn.Module):
    """
        Multi-Layer Perceptron
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Conv1dMlp(nn.Module):
    """
        Multi-Layer Perceptron
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        conv_kernel_size: int = 5,
        conv_padding: int = 2,
        # pool_kernel_size: int = 2,
        # pool_padding: int = 0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Conv1d(in_features, hidden_features, kernel_size=conv_kernel_size, stride=1, padding=conv_padding, bias=bias)
        self.act = act_layer()
        self.conv2 = nn.Conv1d(hidden_features, out_features, kernel_size=conv_kernel_size, stride=1, padding=conv_padding, bias=bias)
        self.drop = nn.Dropout(drop)
        # self.maxpool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size, padding=pool_padding)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        # x = self.maxpool(x)
        return x
    
class SelfAttention(nn.Module):
    """
        Multi-Head Self Attn
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3,B,h,N,d

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class CrossAttention(nn.Module):
    """
        Multi-Head Cross Attn
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 2,B,h,N,d
        q = self.q(y).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 1,B,h,N,d

        q = q[0] * self.scale
        k, v = kv[0], kv[1]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class SA(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.sa = SelfAttention(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class CA(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln1_x = nn.LayerNorm(dim)
        self.ln1_y = nn.LayerNorm(dim)
        self.ca = CrossAttention(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x + self.ca(self.ln1_x(x), self.ln1_y(y))
        x = x + self.mlp(self.ln2(x))
        return x
