from observe_policy import ObservePolicy,Observation
from composition_gap_detector import CompositionGapDetector
from exploration_gates import CandidateEvidence,existence_gate,promotion_gate
from research_scheduler import ResearchScheduler

def test_observe_has_finite_exit():
 p=ObservePolicy(min_replicates=3,max_generations=5)
 h=[Observation(1,.02,-.01,.05),Observation(3,.02,-.01,.05),Observation(5,.02,-.01,.05)]
 assert p.decide(h)=="LIMITED"

def test_observe_promotes_only_clear_signal():
 p=ObservePolicy()
 h=[Observation(1,.1,.01,.2),Observation(2,.1,.02,.2),Observation(3,.1,.03,.2)]
 assert p.decide(h)=="SURVIVE"

def test_composition_gap_is_distinct_from_component_gap():
 d=CompositionGapDetector(min_obs=3)
 for _ in range(3):d.observe("a+b",{"a":.9,"b":.85},.4)
 assert d.gaps()[0]["composition"]=="a+b"

def test_existence_does_not_imply_promotion():
 e=CandidateEvidence(False,.8,.2,.4,.4,.4,.4)
 assert existence_gate(e)
 assert not promotion_gate(e,{"quality":.7})

def test_scheduler_preserves_three_budget_modes():
 s=ResearchScheduler(["a","b","c"],seed=1)
 assert s.shares==(.70,.20,.10)
