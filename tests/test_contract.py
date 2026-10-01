import torch
from symcore import compress, decompress

def _roundtrip(x, symmetry_types):
    xc, pos = compress(x, window_size=16, epsilon=0.0, symmetry_types=symmetry_types)
    xr = decompress(xc, pos, x.shape[1])
    return xc, xr

def test_exact_periodic_roundtrip():
    torch.manual_seed(101)
    base = torch.randn(2, 4, 8)
    x = base.repeat(1, 16, 1)
    _, xr = _roundtrip(x, ["periodic"])
    assert torch.equal(x, xr)

def test_exact_mirror_roundtrip():
    torch.manual_seed(102)
    half = torch.randn(2, 32, 8)
    x = torch.cat([half, torch.flip(half, dims=[1])], dim=1)
    _, xr = _roundtrip(x, ["mirror"])
    assert torch.equal(x, xr)

def test_exact_scale_roundtrip():
    torch.manual_seed(103)
    first = torch.randn(2, 8, 8)
    window = torch.cat([first, first * 2.0], dim=1)
    x = window.repeat(1, 4, 1)
    _, xr = _roundtrip(x, ["scale"])
    assert torch.equal(x, xr)

def test_no_symmetry_is_lossless():
    torch.manual_seed(104)
    x = torch.randn(2, 64, 8)
    _, xr = _roundtrip(x, [])
    assert torch.equal(x, xr)
