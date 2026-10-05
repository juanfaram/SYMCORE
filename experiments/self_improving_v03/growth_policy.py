#!/usr/bin/env python3
"""Pi_0 -> Pi_1: first operational growth-policy learner over causal HostState.
Pi_0 is the frozen heuristic controller; Pi_1 learns intervention value by host-state bins.
No autonomous code modification or gate modification."""
from collections import defaultdict
class GrowthPolicy:
 def __init__(self,min_support=20,margin=0.0):
  self.min_support=min_support;self.margin=margin;self.stats=defaultdict(lambda:[0,0.0,0.0])
 def key(self,s):
  return (int(s["error_velocity"]>0),int(s["change_probability"]>.6),int(s["recent_regret"]>0),int(s["expert_disagreement"]>0))
 def observe(self,state,intervention_gain,risk_violation=False):
  x=self.stats[self.key(state)];x[0]+=1;x[1]+=float(intervention_gain);x[2]+=float(bool(risk_violation))
 def score(self,state):
  n,g,b=self.stats[self.key(state)]
  if n<self.min_support:return None
  return {"n":n,"expected_gain":g/n,"risk_rate":b/n}
 def decide(self,state,risk_budget=.05):
  z=self.score(state)
  return bool(z and z["expected_gain"]>self.margin and z["risk_rate"]<risk_budget)
