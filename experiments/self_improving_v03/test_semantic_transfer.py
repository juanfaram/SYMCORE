from acquisition_state import AcquisitionState
from acquisition_experience import AcquisitionExperience
from intervention_effects import INTENTS,Effect
from effect_adapter import EffectAdapter
from semantic_growth_policy import SemanticGrowthPolicy
def test_semantics_cross_local_names():
 a=EffectAdapter({"x":Effect(plasticity=1,exploration=.7),"y":Effect(stability=1)})
 assert a.decode("adapt_fast")=="x"
 s=AcquisitionState(10,.6,1.2,.3,.1,.1);p=SemanticGrowthPolicy(2)
 p.observe(s,"adapt_fast",1);p.observe(s,"adapt_fast",1)
 assert p.choose(s)=="adapt_fast"
 assert AcquisitionExperience(s,INTENTS["adapt_fast"],1,.1,.0,.9).utility()>0
