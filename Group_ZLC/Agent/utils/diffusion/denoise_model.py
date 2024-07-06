import torch
import torch.nn as nn
import numpy as np
import math
from utils.transformer.transformer import Transformer
from utils.diffusion import *
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, total_time_steps=1000, max_len = 45, d_model = 20):
        super(SinusoidalPositionEmbeddings, self).__init__()
        time_emb_dims = max_len * d_model
        time_emb_dims_exp = time_emb_dims
        ########################################################################
        half_dim = time_emb_dims // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

        ts = torch.arange(total_time_steps, dtype=torch.float32)

        emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        self.time_blocks = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(in_features=time_emb_dims, out_features=time_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dims_exp, out_features=time_emb_dims_exp),
        )
        ########################################################################
        self.max_len = max_len
        self.d_model = d_model
    def forward(self, time):
        result = self.time_blocks(time)
        result = result.view(*result.shape[:-1], self.max_len, self.d_model )
        return result
# time_emb_dims: embedding dimensions
# time_emb_dims_exp: expanding dimensions
class Denoise(nn.Module):
    def __init__(self, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, max_time_step = 1000, device = 'cuda'):
        super(Denoise, self).__init__()
        self.model = Transformer(max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device)
        # [s_{t+1}^{k}, r_{k}^{k}, s_{t}, a_{t}]
        self.max_time_step = max_time_step
        self.input = nn.Sequential(
            nn.Linear(d_model, d_model, device=device),
            nn.Linear(d_model, d_model, device=device),
            nn.Linear(d_model, d_model, device=device),
        )
        self.mask = torch.tril(torch.ones(max_len, max_len)).type(torch.ByteTensor).to(device)
        self.time_emb = SinusoidalPositionEmbeddings(max_time_step, max_len, d_model)
        self.time_emb = self.time_emb.to(device)
    def forward(self, x: torch.Tensor, time: torch.Tensor):
        # x: [[s_{t+1}^{k}, r_{k}^{k}, s_{t}, a_{t}],...]
        x = self.input(x) + self.time_emb(time)
        noise = self.model(x, self.mask)
        return noise