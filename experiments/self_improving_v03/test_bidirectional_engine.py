from bidirectional_engine import BidirectionalEngine,EvolutionState,Component
from capability_frontier import Capability,Interval,Frontier
def C(q,a,r,e,l,w=.005):return Capability(*[Interval(x,x-w,x+w) for x in (q,a,r,e,l)])
def test_prune_is_reversible_and_only_commits_after_gate():
 e=BidirectionalEngine(Frontier(regression_budgets={k:.05 for k in ("quality","adaptation","retention","efficiency","learnability")}))
 s=EvolutionState({"a":Component("a"),"redundant":Component("redundant")})
 p=e.propose(s,"prune","redundant");shadow=e.apply_shadow(s,p)
 assert "redundant" in s.components and "redundant" in shadow.archive
 s2,ok=e.decide(s,p,C(.8,.8,.8,.6,.7),C(.8,.8,.8,.75,.76));assert ok and "redundant" in s2.archive
 restore=e.propose(s2,"restore","redundant");s3=e.apply_shadow(s2,restore);assert "redundant" in s3.components
def test_rejected_destructive_change_rolls_back_exactly():
 e=BidirectionalEngine(Frontier(regression_budgets={k:.05 for k in ("quality","adaptation","retention","efficiency","learnability")}))
 s=EvolutionState({"core":Component("core",{"x":1})});p=e.propose(s,"prune","core")
 out,ok=e.decide(s,p,C(.8,.8,.8,.7,.75),C(.9,.8,.4,.9,.8))
 assert not ok and out==s and out.version==0 and "core" in out.components
def test_all_operations_have_shadow_semantics():
 s=EvolutionState({"a":Component("a",{"x":1}),"b":Component("b",{"y":2})});e=BidirectionalEngine()
 cases=[("create","c",{"z":3}),("modify","a",{"x":2}),("combine","ab",{"sources":["a","b"]}),("freeze","a",{}),("prune","b",{}),("noop",None,{})]
 for op,t,payload in cases:
  q=e.propose(s,op,t,payload);shadow=e.apply_shadow(s,q);assert shadow.version==1 and s.version==0
