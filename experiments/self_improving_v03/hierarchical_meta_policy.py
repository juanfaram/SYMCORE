#!/usr/bin/env python3
"""Hierarchical meta-policy: choose evolutionary operation then its parameters from causal predictions."""
from __future__ import annotations
from dataclasses import dataclass
from causal_evolution_memory import AXES
@dataclass(frozen=True)
class Proposal:
 operation:str;params:tuple;score:float;robust_delta:tuple;cost_ucb:float
class HierarchicalMetaPolicy:
 def __init__(self,memory,risk_budget=.05,evidence_price=.0):
  self.memory=memory;self.risk_budget=risk_budget;self.evidence_price=evidence_price
 def evaluate(self,z,effect,cost_floor=1e-6):
  p=self.memory.predict(z,effect)
  robust=tuple(p.means[a]-p.ci95[a] for a in AXES)
  if min(robust)<-self.risk_budget:return -1e12,robust,p
  cost=max(cost_floor,p.cost_mean+p.cost_ci95)
  return sum(max(0.,x) for x in robust)/cost-self.evidence_price*sum(p.ci95.values()),robust,p
 def choose(self,z,space):
  # space: operation -> iterable[(params,effect)]. NOOP is an ordinary candidate with empty params/effect.
  best=None
  for op,candidates in space.items():
   for params,effect in candidates:
    score,robust,p=self.evaluate(z,effect)
    q=Proposal(op,tuple(params),score,robust,p.cost_mean+p.cost_ci95)
    if best is None or q.score>best.score:best=q
  if best is None:raise ValueError("empty proposal space")
  return best
