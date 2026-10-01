import random
from omega.discovery01.datasets import make_case
from omega.discovery01.oracles import recompute_reference, oracle_optimized_recompute
from omega.discovery01.realization import discovered_realization

def test_adversarial():
    cases=[
      ([],[("ADD",0),("REMOVE",0)]),
      ([2,2],[("REMOVE",2),("ADD",-2),("REMOVE",2)]),
      ([-3,0,3],[("ADD",10**12),("REMOVE",0),("ADD",-10**12)])
    ]
    for s,ops in cases:
        exp=recompute_reference(s,ops)
        assert discovered_realization(s,ops)==exp
        assert oracle_optimized_recompute(s,ops)==exp

def test_1000_differential_cases():
    for seed in range(4300,5300):
        s,ops=make_case(seed,seed%40,1+seed%80)
        assert discovered_realization(s,ops)==recompute_reference(s,ops)

def test_property_add_remove_restores_observable():
    rng=random.Random(4401)
    for _ in range(500):
        s=[rng.randrange(-100,101) for _ in range(rng.randrange(0,30))]
        x=rng.randrange(-100,101)
        ops=[("ADD",x),("REMOVE",x)]
        out=discovered_realization(s,ops)
        assert out[-1]==sum(v*v for v in s)
