import torch
from .utils import compute_norm, get_divisors

def _row_norms(x: torch.Tensor, norm_type: str) -> torch.Tensor:
    if norm_type == 'l2':
        return torch.linalg.vector_norm(x, ord=2, dim=-1)
    if norm_type == 'l1':
        return torch.linalg.vector_norm(x, ord=1, dim=-1)
    if norm_type == 'linf':
        return torch.linalg.vector_norm(x, ord=float('inf'), dim=-1)
    raise ValueError(f"Unknown norm type: {norm_type}")

def detect_symmetry(window: torch.Tensor, epsilon: float,
                    symmetry_types: list, norm_type: str, config):
    k = window.shape[0]

    if 'mirror' in symmetry_types:
        half = k // 2
        if half:
            left = window[:half]
            right = torch.flip(window[k-half:], dims=[0])
            if bool(torch.all(_row_norms(left - right, norm_type) < epsilon)):
                return 'mirror', {}

    if 'periodic' in symmetry_types:
        for p in get_divisors(k):
            if p > k * config.get('max_period_ratio', 0.5):
                continue
            if bool(torch.all(_row_norms(window[:-p] - window[p:], norm_type) < epsilon)):
                return 'periodic', {'period': p, 'reps': k // p}

    if 'scale' in symmetry_types and k % 2 == 0:
        half = k // 2
        alpha_estimates = []
        for m in range(half):
            norm_a = compute_norm(window[m], norm_type)
            if norm_a > epsilon:
                norm_b = compute_norm(window[half+m], norm_type)
                alpha_estimates.append((norm_b / norm_a).item())
        if alpha_estimates:
            alpha = torch.tensor(alpha_estimates).median().item()
            is_scale = True
            for m in range(half):
                if compute_norm(window[half+m] - alpha * window[m], norm_type) >= epsilon:
                    is_scale = False
                    break
            if is_scale:
                return 'scale', {'factor': alpha}

    return None, {}
