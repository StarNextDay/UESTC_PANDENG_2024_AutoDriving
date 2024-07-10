from typing import Callable, Optional, List
import torch
from torch import nn

from layers.OurAttention import *
import sys
from .iTransformer import Model as iTransformer

class OurModel(nn.Module):
    def __init__(self, args,) -> None:
        super().__init__()
        self.tasks = args.tasks
        self.n_tasks = len(self.tasks)
        self.encoder = iTransformer(args)
        self.dim = args.d_model #!

        self.cls_tokens = nn.ModuleDict({i: nn.Parameter(torch.zeros(1,1,self.dim)) 
                                         for i in range(self.n_tasks + 1)})
        self.our_blks = nn.ModuleList([OurBlk(self.dim, self.n_tasks) 
                                       for i in range(args.depth)])
        self.task_heads = nn.ModuleDict({i: nn.Linear(self.dim, self.tasks[i])
                                         for i in range(self.n_tasks)})

        
    def forward_features(self, x, softmax=True) -> torch.Tensor:
        embedding = self.encoder.encode(x) #! B,N,C

        # duplicate n+1 copies & add cls tokens
        x = torch.stack([torch.cat([self.cls_tokens[i],embedding],dim=1) for i in range(self.n_tasks+1)]) #! T+1,B,N+1,C

        for depth, blk in enumerate(self.our_blks):
            x = blk(x)

        # T+1,B,N+1,C -> T+1,B,1,C -> T,B,1,C -> T,B,C
        x = x[:-1,:,0,:]

        predictions = {}
        for i in range(self.n_tasks):
            prediction = self.task_heads[i](x[i]) # B,C -> B,C'
            if softmax:
                predictions[i] = prediction if self.tasks[i]==1 else prediction.softmax(-1)
            else:
                predictions[i] = prediction

        return predictions
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
