import torch
import torch.nn as nn
from typing import List, Tuple
from dataclasses import dataclass

from .detection import detect_symmetry
from .collapse import collapse_window
from .utils import validate_input

@dataclass
class SymCoreConfig:
    window_size: int = 16
    epsilon: float = 0.01
    symmetry_types: Tuple[str, ...] = ('periodic', 'mirror', 'scale')
    norm_type: str = 'l2'
    max_period_ratio: float = 0.5
    
    def validate(self):
        if not 4 <= self.window_size <= 256:
            raise ValueError(f"window_size must be in [4, 256]")
        if not 1e-6 <= self.epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [1e-6, 1.0]")

def compress(X: torch.Tensor, window_size: int = 16, epsilon: float = 0.01,
              symmetry_types: List[str] = ['periodic', 'mirror', 'scale'],
              norm_type: str = 'l2', return_metadata: bool = True):
    
    validate_input(X)
    config = {'max_period_ratio': 0.5}
    
    batch_size, seq_len, d_model = X.shape
    device = X.device
    
    compressed_batches = []
    position_maps = []
    
    for b in range(batch_size):
        x = X[b]
        compressed = []
        pos_map = []
        i = 0
        
        while i < seq_len:
            win_len = min(window_size, seq_len - i)
            window = x[i:i+win_len]
            
            sym_type, params = detect_symmetry(
                window, epsilon, symmetry_types, norm_type, config
            )
            
            if sym_type is not None and win_len == window_size:
                collapsed, meta = collapse_window(window, sym_type, params)
                compressed.extend([c.unsqueeze(0) for c in collapsed])
                pos_map.append({
                    'original_start': i, 'original_end': i + window_size - 1,
                    'type': sym_type, 'metadata': meta
                })
                i += window_size
            else:
                compressed.append(window[0:1])
                pos_map.append({
                    'original_start': i, 'original_end': i,
                    'type': 'none', 'metadata': {}
                })
                i += 1
        
        X_c = torch.cat(compressed, dim=0)
        compressed_batches.append(X_c)
        position_maps.append(pos_map)
    
    max_len = max(c.shape[0] for c in compressed_batches)
    X_padded = torch.zeros(batch_size, max_len, d_model, device=device, dtype=X.dtype)
    for b, c in enumerate(compressed_batches):
        X_padded[b, :c.shape[0], :] = c
    
    return X_padded, position_maps

def decompress(X_compressed, position_map, original_length, fill_value=0.0):
    batch_size = X_compressed.shape[0]
    d_model = X_compressed.shape[2]
    device = X_compressed.device
    dtype = X_compressed.dtype
    
    X_reconstructed = torch.full(
        (batch_size, original_length, d_model), fill_value, device=device, dtype=dtype
    )
    
    for b in range(batch_size):
        x_c = X_compressed[b]
        pos_map = position_map[b]
        coll_idx = 0
        
        for entry in pos_map:
            if entry['type'] == 'none':
                X_reconstructed[b, entry['original_start']] = x_c[coll_idx]
                coll_idx += 1
            else:
                if entry['type'] == 'mirror':
                    k = entry['original_end'] - entry['original_start'] + 1
                    base_len = (k + 1) // 2
                    base = x_c[coll_idx:coll_idx+base_len]
                    coll_idx += base_len
                    mirrored = torch.flip(base[:k//2], dims=[0])
                    full = torch.cat([base, mirrored], dim=0)[:k]
                    X_reconstructed[b, entry['original_start']:entry['original_end']+1] = full
                
                elif entry['type'] == 'periodic':
                    p = entry['metadata']['period']
                    reps = entry['metadata']['reps']
                    base = x_c[coll_idx:coll_idx+p]
                    coll_idx += p
                    full = base.repeat(reps, 1)
                    X_reconstructed[b, entry['original_start']:entry['original_end']+1] = full
                
                elif entry['type'] == 'scale':
                    k = entry['original_end'] - entry['original_start'] + 1
                    half = k // 2
                    alpha = entry['metadata']['factor']
                    base = x_c[coll_idx:coll_idx+half]
                    coll_idx += half
                    scaled = base * alpha
                    full = torch.cat([base, scaled], dim=0)
                    X_reconstructed[b, entry['original_start']:entry['original_end']+1] = full
    
    return X_reconstructed

class SymCoreLayer(nn.Module):
    def __init__(self, window_size=16, epsilon=0.01, 
                 symmetry_types=['periodic', 'mirror', 'scale']):
        super().__init__()
        self.window_size = window_size
        self.register_buffer('epsilon', torch.tensor(epsilon))
        self.symmetry_types = symmetry_types
    
    def forward(self, x):
        return compress(x, self.window_size, self.epsilon.item(), self.symmetry_types)