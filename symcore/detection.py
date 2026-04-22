import torch
from .utils import compute_norm, get_divisors

def detect_symmetry(window: torch.Tensor, epsilon: float, 
                    symmetry_types: list, norm_type: str, config):
    k = window.shape[0]
    
    if 'mirror' in symmetry_types:
        is_mirror = True
        for m in range(k // 2):
            if compute_norm(window[m] - window[k-1-m], norm_type) >= epsilon:
                is_mirror = False
                break
        if is_mirror:
            return 'mirror', {}
    
    if 'periodic' in symmetry_types:
        for p in get_divisors(k):
            if p > k * config.get('max_period_ratio', 0.5):
                continue
            is_periodic = True
            for m in range(k - p):
                if compute_norm(window[m] - window[m+p], norm_type) >= epsilon:
                    is_periodic = False
                    break
            if is_periodic:
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