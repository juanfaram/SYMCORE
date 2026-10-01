import torch
from symcore import compress, decompress

EPS_EXACT = 1e-6

def _roundtrip(x, symmetry_types):
    xc, pos = compress(x, window_size=16, epsilon=EPS_EXACT, symmetry_types=symmetry_types)
    xr = decompress(xc, pos, x.shape[1])
    return xc, pos, xr

def _assert_detected_and_lossless(x, types):
    xc,pos,xr=_roundtrip(x,types)
    assert xc.shape[1] < x.shape[1], "round-trip alone is insufficient: symmetry must actually compress"
    assert any(e["type"] != "none" for batch in pos for e in batch)
    assert torch.equal(x,xr)

def test_exact_periodic_detects_compresses_and_roundtrips():
    torch.manual_seed(101)
    base=torch.randn(2,4,8); x=base.repeat(1,16,1)
    _assert_detected_and_lossless(x,["periodic"])

def test_exact_mirror_detects_compresses_and_roundtrips():
    torch.manual_seed(102)
    block_half=torch.randn(2,8,8)
    block=torch.cat([block_half,torch.flip(block_half,dims=[1])],dim=1)
    x=block.repeat(1,4,1)
    _assert_detected_and_lossless(x,["mirror"])

def test_exact_scale_detects_compresses_and_roundtrips():
    torch.manual_seed(103)
    first=torch.randn(2,8,8)
    block=torch.cat([first,first*2.0],dim=1)
    x=block.repeat(1,4,1)
    _assert_detected_and_lossless(x,["scale"])

def test_no_symmetry_is_lossless():
    torch.manual_seed(104)
    x=torch.randn(2,64,8)
    xc,pos=compress(x,window_size=16,epsilon=EPS_EXACT,symmetry_types=[])
    xr=decompress(xc,pos,x.shape[1])
    assert torch.equal(x,xr)
    assert xc.shape[1] == x.shape[1]
