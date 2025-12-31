import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from .base import BaseScheduler, BaseModel

class FlowMatchingScheduler(BaseScheduler):

    def __init__(self, timesteps: int, **kwargs):
        super().__init__(num_train_timesteps=timesteps, **kwargs)
        
    def sample_time(self, batch_size: int) -> torch.Tensor:
        # Uniform in [0, 1]
        return torch.rand(batch_size, device=self.device)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, **kwargs) -> torch.Tensor:
        
        # Linear interpolation between data x (t=0) and noise z (t=1):
        # x_t = (1 - s) * x + s * z
        if t.dtype.is_floating_point():
            s = torch.clamp(t, 0.0, 1.0)
        else:
            denom = max(self.num_train_timesteps - 1, 1)
            s = t.float() / float(denom)
        while s.dim() < x.dim():
            s = s.unsqueeze(-1)
        return (1.0 - s) * x + s * noise
    
    def reverse(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        
        dt = kwargs.get("dt", None)
        if dt is None:
            raise ValueError("reverse step requires 'dt' in kwargs")
        if "velocity" in kwargs and kwargs["velocity"] is not None:
            v = kwargs["velocity"]
        else:
            model = kwargs.get("model", None)
            if model is None:
                raise ValueError("reverse requires either 'velocity' or 'model'")
            v = model.predict(x, t, **{k: v for k, v in kwargs.items() if k not in ("dt", "model")})
        return x - dt * v
    
class FlowMatchingModel(BaseModel):

    def __init__(self, model: nn.Module, scheduler: BaseScheduler, **kwargs):
        super().__init__(model, scheduler, **kwargs)

    def predict(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        context = kwargs.get("context", None)
        return self.model(x, t, context)

    def loss(
        self,
        x: torch.Tensor,                    # clean data x0
        noise: torch.Tensor,                # noise z ~ N(0, I)
        t: Optional[torch.Tensor] = None,   # times in [0,1] or integer steps
        **kwargs,
    ) -> torch.Tensor:
        B = x.shape[0]
        device = x.device
        if t is None:
            t = self.scheduler.sample_time(B).to(device)

        x_t = self.scheduler.forward(x, t, noise)
        v_target = noise - x

        v_pred = self.predict(x_t, t, **kwargs)
        return torch.mean((v_pred - v_target) ** 2)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        **kwargs,
    ) -> torch.Tensor:
        
        shape = kwargs.get("shape", None)
        if shape is None:
            raise ValueError("sample requires 'shape' (B,C,H,W) or (C,H,W)")
        if len(shape) == 3:
            C, H, W = shape
            shape = (batch_size, C, H, W)
        elif len(shape) == 4:
            if shape[0] != batch_size:
                raise ValueError("shape[0] must equal batch_size")
        else:
            raise ValueError("shape must be length 3 or 4")

        context = kwargs.get("context", None)
        if context is None:
            raise ValueError("sample requires 'context'")
        device = self.scheduler.device

        steps = int(kwargs.get("num_inference_steps", 50))
        assert steps >= 1
        generator = kwargs.get("generator", None)

        x = torch.randn(shape, device=device, generator=generator)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

        for i in range(steps):
            t_i = ts[i].expand(batch_size)
            dt = (ts[i] - ts[i + 1]).abs() # positive
            v = self.predict(x, t_i, context=context)
            x = x - dt * v
        return x
