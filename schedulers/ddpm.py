import math
from typing import Optional

import torch
from .base import BaseScheduler


class DDPMScheduler(BaseScheduler):
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        beta_schedule: str = "linear",  # or "cosine"
        prediction_type: str = "epsilon",
        clip_x0: bool = True,
        clip_range: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            prediction_type=prediction_type,
            clip_x0=clip_x0,
            clip_range=clip_range,
            device=device,
        )

        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, self.num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "cosine":
            betas = betas_for_alpha_bar(self.num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        self.register(betas)
        self.set_timesteps(self.num_train_timesteps)

    def register(self, betas: torch.Tensor) -> None:
        super().register(betas)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # clip for numerical stability
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device: torch.device):
        super().to(device)
        for name in [
            "posterior_variance",
            "posterior_log_variance_clipped",
            "posterior_mean_coef1",
            "posterior_mean_coef2",
        ]:
            tensor = getattr(self, name)
            setattr(self, name, tensor.to(device))
        return self

    def set_timesteps(self, num_inference_steps: int) -> None:
        super().set_timesteps(num_inference_steps)

    @staticmethod
    def extract(a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return BaseScheduler.extract(a, t, x)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return super().add_noise(x0, noise, t)

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return self.extract(self.sqrt_recip_alphas_cumprod, t, x_t) * x_t - \
               self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t) * eps

    def step(
        self,
        model_output: torch.Tensor,  # predicted noise (eps)
        t: torch.Tensor,             # (B,) long timesteps
        x_t: torch.Tensor,           # current sample
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
       
        # compute model mean of q(x_{t-1} | x_t, x0)
        x0_pred = self.compute_x0(model_output, t, x_t)
        if self.clip_x0:
            x0_pred = torch.clamp(x0_pred, -self.clip_range, self.clip_range)

        coef1 = self.extract(self.posterior_mean_coef1, t, x_t)
        coef2 = self.extract(self.posterior_mean_coef2, t, x_t)
        model_mean = coef1 * x0_pred + coef2 * x_t

        # variance noise except for t == 0
        log_var = self.extract(self.posterior_log_variance_clipped, t, x_t)
        noise = torch.randn_like(x_t, generator=generator)

        nonzero_mask = (t != 0).to(x_t.dtype)
        while nonzero_mask.dim() < x_t.dim():
            nonzero_mask = nonzero_mask.unsqueeze(-1)
        x_prev = model_mean + nonzero_mask * torch.exp(0.5 * log_var) * noise
        return x_prev

def betas_for_alpha_bar(num_steps: int, max_beta: float = 0.999) -> torch.Tensor:
   
    def alpha_bar(t):
        s = 0.008
        return (
            math.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        )

    betas = []
    for i in range(num_steps):
        t1 = i / num_steps
        t2 = (i + 1) / num_steps
        beta = min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta)
        betas.append(beta)
    return torch.tensor(betas, dtype=torch.float32)