import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, output_dim: int, embed_dim: int = 1280, hidden_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        if x.dim() == 0:
            x = x.unsqueeze(0)

        half_dim = self.embed_dim // 2
        device, dtype = x.device, x.dtype

        vals = torch.arange(half_dim, device=device, dtype=dtype)
        scale = torch.exp(-math.log(10000.0) * (2 * vals / self.embed_dim))
        x = x[:, None]  
        scaled_input = x * scale  
        pos_encoding = torch.cat([torch.sin(scaled_input), torch.cos(scaled_input)], dim=-1)

        y = self.fc1(pos_encoding)
        y = self.silu(y)
        y = self.fc2(y)

        return y
    
class SiLU(nn.Module): 
    def __init__(self): 
        super().__init__() 
    
    def forward(self, x: torch.Tensor)-> torch.Tensor: 
        return x / (1.0 + torch.exp(-x))
