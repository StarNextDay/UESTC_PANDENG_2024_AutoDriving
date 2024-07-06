import torch
from torch import nn
from utils.transformer.layer import Layer
from utils.transformer.positional_encoding import PositionalEncoding
class Transformer(nn.Module):
    def __init__(self, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super(Transformer, self).__init__()
        ### Varible Defination:
        # max_len: max sequence len
        # d_model: dimention of model
        # ffn_hidden: dimention of Feed Forward's hidden layer
        # n_head: num of attention heads
        # !!! d_model % n_head = 0
        # n_layers: num of layers
        # drop_prob: drop probability
        # device: 'cuda'/'cpu'
        self.emb = PositionalEncoding(d_model=d_model,
                                        max_len=max_len,
                                        device=device)
        self.layers = nn.ModuleList([Layer(d_model=d_model,
                                            ffn_hidden=ffn_hidden,
                                            n_head=n_head,
                                            drop_prob=drop_prob,
                                            device=device)
                                     for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, d_model).to(device)
    def forward(self, trg, mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, mask)

        # pass to LM head
        output = self.linear(trg)
        return output