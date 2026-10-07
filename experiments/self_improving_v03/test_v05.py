import random
from main import Genome
from meta_evolution import EvolutionMemory,apply
from seasonal import SeasonalMemory
def test_seasonal_learns():
    m=SeasonalMemory(("hr",),4);r={"hr":"8"}
    assert m.predict(r)==0
    for y in (10,20,30,40):m.learn(r,y)
    assert m.predict(r)==25
def test_meta_operator_bounds(tmp_path):
    g=Genome()
    for op in ("lr_up","lr_down","toggle_features","momentum","regularize","clip"):
        x=apply(g,op,random.Random(2));assert .002<=x.lr<=1 and 0<=x.momentum<=.95 and 50<=x.clip<=1000
def test_meta_memory_prefers_rewarded_operator(tmp_path):
    m=EvolutionMemory(tmp_path/"m.json",1)
    for op in ("lr_up","lr_down","toggle_features","momentum","regularize","clip"):m.record("stable",op,-1)
    for _ in range(30):m.record("stable","lr_down",1)
    picks=[m.choose("stable") for _ in range(20)]
    assert picks.count("lr_down")>=15
