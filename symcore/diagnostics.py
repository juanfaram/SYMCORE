import torch
import time
import numpy as np
from .core import compress

def analyze_symmetry_density(X, window_size=16, epsilon=0.01):
    if X.dim() == 2:
        X = X.unsqueeze(0)
    from .detection import detect_symmetry
    
    config = {'max_period_ratio': 0.5}
    total_windows = 0
    symmetric_windows = 0
    type_counts = {'mirror': 0, 'periodic': 0, 'scale': 0, 'none': 0}
    
    for b in range(X.shape[0]):
        seq = X[b]
        seq_len = seq.shape[0]
        for i in range(seq_len - window_size + 1):
            window = seq[i:i+window_size]
            sym_type, _ = detect_symmetry(
                window, epsilon, ['periodic', 'mirror', 'scale'], 'l2', config
            )
            total_windows += 1
            if sym_type is not None:
                symmetric_windows += 1
                type_counts[sym_type] += 1
            else:
                type_counts['none'] += 1
    
    density = symmetric_windows / total_windows if total_windows > 0 else 0
    return {
        'density': density,
        'total_windows': total_windows,
        'symmetric_windows': symmetric_windows,
        'type_distribution': type_counts,
        'estimated_r_lower_bound': 1 + density * (window_size - 2) / 3
    }

def benchmark_symcore(X, config, n_iterations=100, warmup=10):
    device = X.device
    for _ in range(warmup):
        compress(X, **config)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    latencies = []
    compression_ratios = []
    
    for _ in range(n_iterations):
        start = time.perf_counter()
        X_c, _ = compress(X, **config)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
        compression_ratios.append(X.shape[1] / X_c.shape[1])
    
    return {
        'mean_latency_ms': np.mean(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p99_latency_ms': np.percentile(latencies, 99),
        'mean_compression_ratio': np.mean(compression_ratios),
    }