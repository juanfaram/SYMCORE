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
        first = window[:half]
        second = window[half:]
        norm_a = _row_norms(first, norm_type)
        valid = norm_a > epsilon
        if bool(torch.any(valid)):
            norm_b = _row_norms(second, norm_type)
            alpha_t = torch.median(norm_b[valid] / norm_a[valid])
            residual = _row_norms(second - alpha_t * first, norm_type)
            if bool(torch.all(residual < epsilon)):
                return 'scale', {'factor': alpha_t.item()}

    return None, {}
