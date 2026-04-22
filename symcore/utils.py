import torch
from typing import List

def compute_norm(x: torch.Tensor, norm_type: str = 'l2') -> torch.Tensor:
    if norm_type == 'l2':
        return torch.norm(x, p=2)
    elif norm_type == 'l1':
        return torch.norm(x, p=1)
    elif norm_type == 'linf':
        return torch.norm(x, p=float('inf'))
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")

def get_divisors(n: int) -> List[int]:
    return [i for i in range(1, n // 2 + 1) if n % i == 0]

def validate_input(X: torch.Tensor) -> None:
    if X.dim() != 3:
        raise ValueError(f"Input must be 3D (batch, seq_len, d_model), got shape {X.shape}")