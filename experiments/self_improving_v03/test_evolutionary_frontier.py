from capability_frontier import Interval,Capability,Frontier
def C(q,a,r,e,l,w=.01):
 return Capability(*[Interval(x,x-w,x+w) for x in (q,a,r,e,l)])
def test_robust_dominance_requires_evidence_not_mean_noise():
 a=C(.80,.80,.80,.80,.80,.01);b=C(.805,.80,.80,.80,.80,.01)
 assert not b.robustly_dominates(a)
 strong=C(.85,.80,.80,.80,.80,.01)
 assert strong.robustly_dominates(a)
def test_pruning_can_expand_frontier_without_more_capacity():
 base=C(.80,.75,.82,.55,.70,.005)
 pruned=C(.80,.75,.82,.72,.76,.005)
 assert pruned.robustly_dominates(base)
def test_tradeoff_survives_without_scalar_weights():
 quality=C(.88,.72,.80,.55,.68,.005);efficient=C(.80,.72,.80,.82,.74,.005)
 f=Frontier()
 assert f.consider("quality",quality);assert f.consider("efficient",efficient)
 assert {n for n,_ in f.items}=={"quality","efficient"}
def test_noop_beats_unsafe_change_via_regression_budget():
 base=C(.80,.80,.80,.70,.75,.005);unsafe=C(.95,.82,.50,.85,.80,.005)
 f=Frontier(regression_budgets={"quality":.05,"adaptation":.05,"retention":.05,"efficiency":.05,"learnability":.05})
 assert not f.consider("unsafe",unsafe,baseline=base)
 assert f.consider("noop",base,baseline=base)
