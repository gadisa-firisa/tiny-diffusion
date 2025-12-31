import torch
import torch.nn as nn
from typing import Optional


class BaseScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        prediction_type: str = "epsilon", 
        clip_x0: bool = True,
        clip_range: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        self.prediction_type = prediction_type
        self.clip_x0 = clip_x0
        self.clip_range = clip_range
        self.device = device or torch.device("cpu")

    def register(self, betas: torch.Tensor) -> None:
        betas = betas.to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=betas.device, dtype=betas.dtype), alphas_cumprod[:-1]]
        )

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

    def to(self, device: torch.device):
        self.device = device
        for name in [
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas_cumprod",
            "sqrt_recipm1_alphas_cumprod",
        ]:
            tensor = getattr(self, name)
            setattr(self, name, tensor.to(device))

        if hasattr(self, "timesteps"):
            self.timesteps = self.timesteps.to(device)
        return self

    def set_timesteps(self, num_inference_steps: int) -> None:
        num_inference_steps = int(num_inference_steps)
        if num_inference_steps <= 0 or num_inference_steps > self.num_train_timesteps:
            raise ValueError("num_inference_steps must be in (0, num_train_timesteps]")
        step_ratio = max(1, self.num_train_timesteps // num_inference_steps)
        timesteps = torch.arange(0, self.num_train_timesteps, step_ratio, dtype=torch.long)
        timesteps = timesteps[-num_inference_steps:]
        self.timesteps = torch.flip(timesteps, dims=[0]).to(self.device)

    @staticmethod
    def extract(a: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = a.gather(0, t)
        while out.dim() < x.dim():
            out = out.unsqueeze(-1)
        return out.to(dtype=x.dtype)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self.extract(self.sqrt_alphas_cumprod, t, x0)
        sqrt_omab = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x0)
        return sqrt_ab * x0 + sqrt_omab * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            self.extract(self.sqrt_recip_alphas_cumprod, t, x_t) * x_t
            - self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t) * eps
        )

    def predict_x0_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        c1 = self.extract(self.sqrt_alphas_cumprod, t, x_t)
        c2 = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_t)
        return c1 * x_t - c2 * v

    def compute_x0(self, model_output: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        pt = self.prediction_type.lower()
        if pt == "epsilon":
            return self.predict_x0_from_eps(x_t, t, model_output)
        if pt in ("x0", "x_0"):
            return model_output
        if pt in ("v_prediction", "v-prediction", "v"):
            return self.predict_x0_from_v(x_t, t, model_output)
        raise NotImplementedError(f"Unknown prediction_type: {self.prediction_type}")


class BaseModel(nn.Module):
    def __init__(self, model: nn.Module, scheduler: BaseScheduler, **kwargs):
        super().__init__()
        self.model = model
        self.scheduler = scheduler

    def predict(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x, t, **kwargs)

    def to(self, device: torch.device):
        super().to(device)
        self.model.to(device)
        self.scheduler.to(device)
        return self
