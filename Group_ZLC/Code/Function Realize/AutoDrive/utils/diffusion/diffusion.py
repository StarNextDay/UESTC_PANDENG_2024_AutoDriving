import torch
class Diffusion:
    def __init__(
        self,
        num_diffusion_timesteps=1000,
        device="cpu",
    ):
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.device = device

        self.initialize()

    def initialize(self):
        # BETAs & ALPHAs required at different places in the Algorithm.
        self.beta  = self.get_betas()
        self.alpha = 1 - self.beta
        
        self.sqrt_beta                       = torch.sqrt(self.beta).to(self.device)
        self.alpha_cumulative                = torch.cumprod(self.alpha, dim=0).to(self.device)
        self.sqrt_alpha_cumulative           = torch.sqrt(self.alpha_cumulative).to(self.device)
        self.one_by_sqrt_alpha               = 1. / torch.sqrt(self.alpha).to(self.device)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative).to(self.device)
         
    def get_betas(self):
        """linear schedule, proposed in original ddpm paper"""
        scale = 1000 / self.num_diffusion_timesteps
        beta_start = scale * 1e-4
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start,
            beta_end,
            self.num_diffusion_timesteps,
            dtype=torch.float32,
            device=self.device,
        )

def forward_diffusion(sd: Diffusion, x0: torch.Tensor, timesteps: int):
    eps = torch.randn_like(x0)  # Noise
    if type(timesteps) == int:
        mean    = sd.sqrt_alpha_cumulative[timesteps] * x0  # Image scaled
        std_dev = sd.sqrt_one_minus_alpha_cumulative[timesteps] # Noise scaled
        sample  = mean + std_dev * eps # scaled inputs * scaled noise
    else:
        sqrt_a_c = sd.sqrt_alpha_cumulative[timesteps]
        sqrt_one_a_c = sd.sqrt_one_minus_alpha_cumulative[timesteps]
        sqrt_a_c = sqrt_a_c.view(len(sqrt_a_c), 1, 1) 
        sqrt_one_a_c = sqrt_one_a_c.view(len(sqrt_one_a_c), 1, 1)
        mean    = sqrt_a_c * x0  # Image scaled
        std_dev = sqrt_one_a_c # Noise scaled
        sample  = mean + std_dev * eps # scaled inputs * scaled noise
    return sample, eps  # return ... , gt noise --> model predicts this)