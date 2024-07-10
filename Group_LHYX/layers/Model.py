"""
author  :   lifangbo
date    :   2024-04-25
"""
from typing import Callable, Optional

import torch
from torch import Tensor
from torch import nn

from .OurAttention import *
from .Transformer_EncDec import Encoder, EncoderLayer
from .SelfAttention_Family import FullAttention, AttentionLayer
from .Embed import DataEmbedding_inverted

class Vanilla_Mlp(nn.Module):
    """
        Vanilla_Mlp Block
    """
    def __init__(self, dim: int, n_tasks: int) -> None:
        super().__init__()
        self.n_tasks = n_tasks

        self.tb = nn.ModuleDict({str(i): Conv1dMlp(dim) for i in range(self.n_tasks)})
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # task_branch Dict[nn.Module]
        

    def forward(self, x: Tensor) -> Tensor:
        """
            CLS & repeat are already done somewhere else
            input T+1,B,N+1,C
        """
        # T, B, N, C = x.shape # task*t, communication*1
        tasks = [] # List[Tensor(B,N,C)]
        for i in range(self.n_tasks):
            task = x[i] + self.tb[str(i)](x[i])
            task = self.maxpool(task)
            tasks.append(task)
        
        return torch.stack(tasks) # task*t, communication*1
    
class Concat_Mlp(nn.Module):
    """
        Concat_Mlp Block
    """
    def __init__(self, dim: int, n_tasks: int) -> None:
        super().__init__()
        self.n_tasks = n_tasks

        self.tb = nn.ModuleDict({str(i): Conv1dMlp(dim * 2, out_features=dim) for i in range(self.n_tasks)})
        # task_branch Dict[nn.Module]
        self.cb = Conv1dMlp(n_tasks * dim, out_features=dim, conv_kernel_size=1, conv_padding=0)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # communication_branch nn.Module

    def forward(self, x: Tensor) -> Tensor:
        """
            CLS & repeat are already done somewhere else
            input T+1,B,N+1,C
        """
        # T, B, N, C = x.shape # task*t, communication*1

        x_com = torch.cat(list(x[:self.n_tasks]),dim=-2)
        communication = self.cb(x_com)
        
        tasks = [] # List[Tensor(B,N,C)]
        for i in range(self.n_tasks):
            task = x[i] + self.tb[str(i)](torch.cat([x[i],communication],dim=-2))
            task = self.maxpool(task)
            tasks.append(task)
        
        # tasks.append(self.maxpool(communication))
        return torch.stack(tasks) # task*t, communication*1

class SA_Blk(nn.Module):
    """
        Our proposed Block
    """
    def __init__(self, dim: int, n_tasks: int) -> None:
        super().__init__()
        self.n_tasks = n_tasks

        self.tb = nn.ModuleDict({str(i): nn.ModuleDict({"sa": SA(dim),}) for i in range(self.n_tasks)}) 
        # task_branch Dict[Dict[nn.Module]]

    def forward(self, x: Tensor) -> Tensor:
        """
            CLS & repeat are already done somewhere else
            input T+1,B,N+1,C
        """
        # T, B, N, C = x.shape # task*t, communication*1
        
        tasks = [] # List[Tensor(B,N,C)]
        for i in range(self.n_tasks):
            task = self.tb[str(i)]["sa"](x[i])
            tasks.append(task)
        
        return torch.stack(tasks) # task*t, communication*1

class Ablation_Blk(nn.Module):
    """
        Our proposed Block
    """
    def __init__(self, dim: int, n_tasks: int) -> None:
        super().__init__()
        self.n_tasks = n_tasks

        self.tb = nn.ModuleDict({str(i): nn.ModuleDict({"sa": SA(dim), "ca": CA(dim),}) for i in range(self.n_tasks)}) 
        # task_branch Dict[Dict[nn.Module]]

    def forward(self, x: Tensor) -> Tensor:
        """
            CLS & repeat are already done somewhere else
            input T+1,B,N+1,C
        """
        # T, B, N, C = x.shape # task*t, communication*1

        communication = torch.zeros_like(x[-1]).detach() #! ablation
        
        tasks = [] # List[Tensor(B,N,C)]
        for i in range(self.n_tasks):
            task = self.tb[str(i)]["sa"](x[i])
            task = self.tb[str(i)]["ca"](task, communication)
            tasks.append(task)
        
        tasks.append(communication)
        return torch.stack(tasks) # task*t, communication*1


class OurBlk(nn.Module):
    """
        Our proposed Block
    """
    def __init__(self, dim: int, n_tasks: int) -> None:
        super().__init__()
        self.n_tasks = n_tasks

        self.tb = nn.ModuleDict({str(i): nn.ModuleDict({"sa": SA(dim), "ca": CA(dim),}) for i in range(self.n_tasks)}) 
        # task_branch Dict[Dict[nn.Module]]
        self.cb = nn.ModuleDict({"sa": SA(dim), "mlp": Mlp(n_tasks * dim, out_features=dim),}) 
        # communication_branch Dict[nn.Module]

    def forward(self, x: Tensor) -> Tensor:
        """
            CLS & repeat are already done somewhere else
            input T+1,B,N+1,C
        """
        T, B, N, C = x.shape # task*t, communication*1

        communication = self.cb["sa"](x[-1])
        
        tasks = [] # List[Tensor(B,N,C)]
        for i in range(self.n_tasks):
            task = self.tb[str(i)]["sa"](x[i])
            task = self.tb[str(i)]["ca"](task, communication)
            tasks.append(task)
        
        communication = self.cb["mlp"](torch.cat(tasks, dim=-1)) # B,N,4*C -> B,N,C
        tasks.append(communication)
        return torch.stack(tasks) # task*t, communication*1
    

class BaseModel(nn.Module):
    def __init__(self, args, ) -> None:
        super().__init__()
        self.tasks = args.tasks
        self.n_tasks = len(self.tasks)
        self.enc_embedding = DataEmbedding_inverted(args.seq_len, args.d_model, args.embed, args.freq,
                                                    args.dropout)
        self.encoder = self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout,
                                      output_attention=args.output_attention), args.d_model, args.n_heads),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation
                ) for l in range(args.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(args.d_model)
        )
        self.dim = args.dim
        self.proj = nn.Linear(args.d_model, self.dim)
        self.cls_tokens = nn.ParameterDict({str(i): nn.Parameter(torch.zeros(1, 1, self.dim))
                                            for i in range(self.n_tasks + 1)})
        self.task_heads = nn.ModuleDict({str(i): Mlp(self.dim, out_features=self.tasks[i])
                                         for i in range(self.n_tasks)})

    def forward_features_(self, x, softmax=False, cls_token=True, transpose=False) -> torch.Tensor:
        batch_size = x.size(0)

        # encode
        x = self.enc_embedding(x, None)
        embedding, _ = self.encoder(x, attn_mask=None)  # ! B,N,C
        embedding = self.proj(embedding)

        # duplicate n+1 copies & add cls tokens
        if cls_token:
            x = torch.stack([torch.cat([self.cls_tokens[str(i)].repeat(batch_size, 1, 1), embedding], dim=1) for i in
                            range(self.n_tasks+1)])  # ! T+1,B,N+1,C
        else:
            x = torch.stack([embedding for i in range(self.n_tasks+1)])  # ! T+1,B,N,C
        
        if transpose:
            x = x.transpose(-2,-1)

        for depth, blk in enumerate(self.our_blks):
            x = blk(x)
        
        if transpose:
            x = x.transpose(-2,-1)

        return x

    def forward(self, x, mask=None, softmax=False):
        x = self.forward_features(x)
        predictions = {}
        for i in range(self.n_tasks):
            prediction = self.task_heads[str(i)](x[i])  # B,C -> B,C'
            if softmax:
                predictions[i] = prediction if self.tasks[i] == 1 else prediction.softmax(-1)
            else:
                predictions[i] = prediction
        return predictions
    

#! baseline
class Baseline_vanilla_mlp(BaseModel):
    #! Baseline
    def __init__(self, args, ) -> None:
        super().__init__(args)
        self.our_blks = nn.ModuleList([Vanilla_Mlp(self.dim, self.n_tasks)
                                       for i in range(args.depth)])
        # self.task_heads = nn.ModuleDict({str(i): nn.Linear(self.dim, self.tasks[i])
        #                                  for i in range(self.n_tasks)})

    def forward_features(self, x, softmax=False) -> torch.Tensor:
        x = self.forward_features_(x, cls_token=False, transpose=True)# T,B,N,C -> T,B,C
        return x.mean(dim=-2)

class Baseline_concat_mlp(BaseModel):
    #! Baseline
    def __init__(self, args, ) -> None:
        super().__init__(args)
        self.our_blks = nn.ModuleList([Concat_Mlp(self.dim, self.n_tasks)
                                       for i in range(args.depth)])
        # self.task_heads = nn.ModuleDict({str(i): nn.Linear(self.dim, self.tasks[i])
        #                                  for i in range(self.n_tasks)})

    def forward_features(self, x, softmax=False) -> torch.Tensor:
        x = self.forward_features_(x, cls_token=False, transpose=True)# T,B,N,C -> T,B,C
        return x.mean(dim=-2)

class Baseline_SA(BaseModel):
    #! Baseline
    def __init__(self, args, ) -> None:
        super().__init__(args)
        self.our_blks = nn.ModuleList([SA_Blk(self.dim, self.n_tasks)
                                       for i in range(args.depth)])
        # self.task_heads = nn.ModuleDict({str(i): nn.Linear(self.dim, self.tasks[i])
        #                                  for i in range(self.n_tasks)})

    def forward_features(self, x, softmax=False) -> torch.Tensor:
        x = self.forward_features_(x)# T,B,1,C -> T,B,C
        return x[:,:,0,:]
    
class Ablation(BaseModel):
    #! Baseline
    def __init__(self, args, ) -> None:
        super().__init__(args)
        self.our_blks = nn.ModuleList([Ablation_Blk(self.dim, self.n_tasks)
                                       for i in range(args.depth)])
        # self.task_heads = nn.ModuleDict({str(i): nn.Linear(self.dim, self.tasks[i])
        #                                  for i in range(self.n_tasks)})

    def forward_features(self, x, softmax=False) -> torch.Tensor:
        x = self.forward_features_(x)
        return x[:,:,0,:]

class OurModel(BaseModel):
    #! Baseline
    def __init__(self, args, ) -> None:
        super().__init__(args)
        self.our_blks = nn.ModuleList([OurBlk(self.dim, self.n_tasks)
                                       for i in range(args.depth)])
        # self.task_heads = nn.ModuleDict({str(i): nn.Linear(self.dim, self.tasks[i])
        #                                  for i in range(self.n_tasks)})

    def forward_features(self, x, softmax=False) -> torch.Tensor:
        x = self.forward_features_(x)
        return x[:,:,0,:]