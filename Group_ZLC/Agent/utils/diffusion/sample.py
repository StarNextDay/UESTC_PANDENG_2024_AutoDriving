import torch
import gc
from utils.diffusion.diffusion import *
from utils.rewarder.rewarder import Rewarder
from utils.diffusion.denoise_model import *

def ddim_sample_discriminator(diffusion: Diffusion, denoise: Denoise, sa: torch.Tensor, rewarder: Rewarder, cur_t: int, sample_steps = 1000, grad_decay = 1e-3):
    with torch.no_grad():
        # with open('data.txt', 'a') as file:
        batch_size = sa.shape[0]
        # time_steps = torch.linspace(0, denoise.max_time_step - 1, denoise.max_time_step / ).__reversed__()
        time_steps = ((np.linspace(0, np.sqrt(denoise.max_time_step - 1), sample_steps)) ** 2).astype(int)
        time_steps = torch.tensor(time_steps, dtype=torch.int32).to(sa.device).__reversed__()
        time_steps = time_steps.to(dtype=torch.int32)
        pred = torch.randn(sa.shape).to(sa.device)
        d_model = sa.shape[-1]
        ### existing problems
    for time_index in range(sample_steps - 1):
        time_step = time_steps[time_index]
        time_next_step = time_steps[time_index + 1]
        if cur_t > 0:
            pred[:,:cur_t-1,:] = sa[:,:cur_t-1,:].clone()
        pred[:,cur_t,:d_model-1] = sa[:,cur_t,:d_model-1].clone()
        grad = rewarder.compute_grad(pred) * grad_decay
        with torch.no_grad():
            time = torch.ones(batch_size, dtype=torch.int32).to(sa.device) * time_step
            time = time.to(dtype=torch.int32)
            ###
            sqrt_alpha = torch.sqrt(diffusion.alpha_cumulative[time_step])
            sqrt_next_alpha = torch.sqrt(diffusion.alpha_cumulative[time_next_step])
            sqrt_one_minus_alpha = torch.sqrt(1 - diffusion.alpha_cumulative[time_step])
            sqrt_one_minus_next_alpha = torch.sqrt(1 - diffusion.alpha_cumulative[time_next_step])
            ###
            pred_noise = denoise(pred, time)
            pred_noise += sqrt_one_minus_alpha * grad
            pred = sqrt_next_alpha * (pred - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha + sqrt_one_minus_next_alpha * pred_noise
            pred = torch.clip(pred, min = -2, max = 2)
            ################################
        if cur_t > 0:
            pred[:,:cur_t-1,:] = sa[:,:cur_t-1,:].clone()
        pred[:,cur_t,:d_model-1] = sa[:,cur_t,:d_model-1].clone()
    torch.cuda.empty_cache()
    gc.collect()
    return pred

class Planning:
    def __init__(self, diffusion: Diffusion, denoise: Denoise, rewarder: Rewarder, max_len, d_model, sample_steps = 10):
        self.diffusion = diffusion
        self.denoise = denoise
        self.rewarder = rewarder
        self.sample_steps = sample_steps
        self.max_len = max_len
        self.d_model = d_model
    def pred(self, sa, cur_t):
        pred = ddim_sample_discriminator(self.diffusion, self.denoise, sa, self.rewarder, cur_t, self.sample_steps)
        if pred.shape[0] == 1:
            a = pred[0, cur_t, self.d_model - 1]
        else:
            a = pred[:, cur_t, self.d_model - 1]
        return pred, a