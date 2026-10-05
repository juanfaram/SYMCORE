import random
from learned_evolution_pi import LearnedEvolutionPi,OPS
def test_policy_learns_contextual_operation_values_without_labels():
 p=LearnedEvolutionPi(kernel_sigma=.2,temperature=.08)
 a=(.9,.8,.1,.2,.0,.7,.2);b=(.1,.3,.9,.9,.0,.2,.4)
 for _ in range(30):
  p.observe(a,"create",1);p.observe(a,"prune",-1);p.observe(b,"create",-1);p.observe(b,"prune",1)
 q=p.frozen()
 assert q.probabilities(a)["create"]>q.probabilities(a)["prune"]
 assert q.probabilities(b)["prune"]>q.probabilities(b)["create"]
def test_frozen_policy_does_not_learn_during_exam():
 p=LearnedEvolutionPi();p.observe((0,)*7,"noop",1);q=p.frozen();n=len(q.memory)
 for i in range(100):q.choose((i/100,)*7,random.Random(i))
 assert len(q.memory)==n
def test_policy_keeps_nonzero_exploration():
 p=LearnedEvolutionPi(temperature=.1)
 for _ in range(20):p.observe((1,)*7,"create",1)
 ps=p.probabilities((1,)*7)
 assert all(ps[o]>0 for o in OPS) and abs(sum(ps.values())-1)<1e-9
