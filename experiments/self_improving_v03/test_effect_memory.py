from effect_memory import EffectMemory
from intervention_effects import Effect
def test_effect_memory_transfers_by_effect_not_name():
 m=EffectMemory();plastic=Effect(plasticity=1,exploration=.5);stable=Effect(stability=1)
 for _ in range(8):m.observe("drift",plastic,1);m.observe("drift",stable,0)
 local={"new_fast_name":Effect(plasticity=.9,exploration=.45),"new_slow_name":Effect(stability=.9)}
 assert m.rank("drift",local)[0]=="new_fast_name"
