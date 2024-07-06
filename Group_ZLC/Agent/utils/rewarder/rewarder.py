import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.transformer.transformer import Transformer

class rewarder(nn.Module):
    def __init__(self, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super(rewarder, self).__init__()
        self.model = Transformer(max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device)
        self.output = nn.Linear(d_model, 1, device = device)
        self.mask = torch.tril(torch.ones(max_len, max_len)).type(torch.ByteTensor).to(device)
    def forward(self, sa):
        reward = self.model(sa, self.mask)
        reward = reward[:, -1, :]
        reward = torch.tanh(self.output(reward))
        return reward

class Rewarder(nn.Module):
    def __init__(self, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device, lr=1e-5):
        super(Rewarder, self).__init__()
        self.model = rewarder(max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device
        self.cnt = 0

    def score(self, sa):
        with torch.no_grad():
            reward = self.model(sa)
        return reward
    
    def grad_score(self, sa):
        reward = self.model(sa)
        return reward

    def compute_grad(self, sa):
        sa = sa.clone().requires_grad_(True)
        reward = self.model(sa)
        reward = - reward
        ### maximize reward
        reward.backward()
        grad = sa.grad
        return grad
    
    def update(self, sa, reward):
        pred_reward = self.model(sa)
        reward = reward.unsqueeze(1)
        loss = F.mse_loss(pred_reward, reward)
        clone_loss = loss.clone().detach().cpu().numpy()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return clone_loss