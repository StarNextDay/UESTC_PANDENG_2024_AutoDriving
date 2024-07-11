import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.diffusion.denoise_model import Denoise
from utils.rewarder.rewarder import Rewarder
from utils.process.dataread import Data
from .diffusion import Diffusion, forward_diffusion
import gc
import os
import random
from tqdm import tqdm
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def CreateBatchData(batch_size, data: Data, sd: Diffusion, max_time_steps = 1000):
    rewards, sa = data.get_datas(batch_size)
    sa = torch.tensor(sa, dtype = torch.float).to(device)
    rewards = torch.tensor(rewards, dtype = torch.float).to(device)
    noise = sa
    timestep = torch.randint(low = 0, high = max_time_steps - 1, size = [batch_size]).to(device)
    noise, target = forward_diffusion(sd, noise, timestep)
    timesteps = torch.ones(batch_size, dtype=torch.int32, device = device) * timestep
    return noise.to(device), target.to(device), timesteps.to(device), sa, rewards
def diffusion_train(epochs, batch_size, save_dir, data:Data, denoise_model: Denoise, rewarder: Rewarder, lr = 1e-3):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sd = Diffusion(num_diffusion_timesteps=denoise_model.max_time_step, device = device)
    optimzer = torch.optim.Adam(denoise_model.parameters(), lr = lr)
    with tqdm(total = epochs) as bar:
        for epoch in range(epochs):
            noise, target, timesteps, sa, rewards = CreateBatchData(batch_size, data, sd, denoise_model.max_time_step)
            pred_noise = denoise_model(noise, timesteps)
            pred_noise[:,:,:2] = pred_noise[:,:,:2] * 100
            target[:,:,:2] = target[:,:,:2] * 100
            loss = F.mse_loss(pred_noise, target)
            rewarder_loss = rewarder.update(sa, rewards)
            bar.set_postfix({'epoch': '%d' % (epoch + 1), 'diffusion loss': '%.3f' % loss.clone().detach().cpu().numpy(), 'rewarder loss': '%.3f' % rewarder_loss})
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
            bar.update(1)
        torch.save(denoise_model, save_dir + "/" + "diffusion.pt")
        torch.save(rewarder, save_dir + "/" + "rewarder.pt")