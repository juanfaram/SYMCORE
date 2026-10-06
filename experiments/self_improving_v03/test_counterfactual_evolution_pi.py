import random
from counterfactual_evolution_pi import CounterfactualEvolutionPi,OPS
def test_counterfactual_policy_prefers_best_relative_action():
 p=CounterfactualEvolutionPi(k=8,temperature=.03)
 a=(.9,.8,.1,.2,.1,.7,.2);b=(.1,.3,.9,.9,.1,.2,.4)
 for _ in range(20):
  p.observe_counterfactual(a,{o:(1 if o=="create" else 0) for o in OPS})
  p.observe_counterfactual(b,{o:(1 if o=="prune" else 0) for o in OPS})
 p.fit()
 assert p.scores(a)["create"]>p.scores(a)["prune"]
 assert p.scores(b)["prune"]>p.scores(b)["create"]
def test_noop_wins_when_interventions_are_worse():
 p=CounterfactualEvolutionPi(k=5,temperature=.02)
 z=(.2,)*7
 for _ in range(10):p.observe_counterfactual(z,{o:(0 if o=="noop" else -.5) for o in OPS})
 p.fit();assert max(p.scores(z),key=p.scores(z).get)=="noop"
def test_frozen_counterfactual_policy_is_read_only_and_explores():
 p=CounterfactualEvolutionPi();z=(.5,)*7;p.observe_counterfactual(z,{o:0 for o in OPS});p.fit();q=p.frozen();n=len(q.rows)
 ps=q.probabilities(z);assert all(v>0 for v in ps.values())
 for i in range(20):q.choose(z,random.Random(i))
 assert len(q.rows)==n
