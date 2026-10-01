import pytest, torch
from symcore import compress, decompress

def roundtrip(x, epsilon=0.0, types=None):
    types = ['periodic','mirror','scale'] if types is None else types
    xc,pos=compress(x,window_size=16,epsilon=epsilon,symmetry_types=types)
    return decompress(xc,pos,x.shape[1])

@pytest.mark.parametrize("length",[1,15,16,17,31,32,33,63,64,65])
def test_random_lengths_lossless_when_detection_disabled(length):
    torch.manual_seed(2000+length)
    x=torch.randn(2,length,8)
    assert torch.equal(x,roundtrip(x,types=[]))

def test_near_mirror_not_exact_at_epsilon_zero():
    torch.manual_seed(3001)
    half=torch.randn(1,8,8)
    other=torch.flip(half,dims=[1]).clone()
    other[0,0,0]+=1e-4
    x=torch.cat([half,other],dim=1)
    xr=roundtrip(x,epsilon=0.0,types=['mirror'])
    assert torch.equal(x,xr)

def test_nan_input_rejected_or_not_claimed():
    # Current contract only supports finite inputs for numerical claims.
    x=torch.zeros(1,16,4); x[0,0,0]=float('nan')
    with pytest.raises((ValueError, AssertionError)):
        compress(x,window_size=16,epsilon=0.0)

def test_inf_input_rejected_or_not_claimed():
    x=torch.zeros(1,16,4); x[0,0,0]=float('inf')
    with pytest.raises((ValueError, AssertionError)):
        compress(x,window_size=16,epsilon=0.0)
