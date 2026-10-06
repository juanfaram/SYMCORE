#!/usr/bin/env python3
"""Strict per-interaction online learner and longitudinal causal instrumentation."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import math
@dataclass(frozen=True)
class Interaction:
 z:tuple;action:str;reward:float;cost:float;new_capability_cost:float
class OnlineMemory:
 def __init__(self):self.n=0;self.stats={};self.total_cost=0.
 def update_one(self,e:Interaction):
  # Exactly one irreversible online update; no replay, dataset, fit(), or batch retraining.
  self.n+=1;self.total_cost+=e.cost;k=(tuple(e.z),e.action);n,s=self.stats.get(k,(0,0.));self.stats[k]=(n+1,s+e.reward)
 def value(self,z,action):
  n,s=self.stats.get((tuple(z),action),(0,0.));return s/n if n else 0.
 def frozen_copy(self):return deepcopy(self)
@dataclass(frozen=True)
class Checkpoint:
 E:int;A:float;C_new:float;live_minus_frozen:float
def slope_ci95(points):
 if len(points)<5:raise ValueError("at least five checkpoints")
 xs=[float(x) for x,_ in points];ys=[float(y) for _,y in points];xm=sum(xs)/len(xs);ym=sum(ys)/len(ys)
 sxx=sum((x-xm)**2 for x in xs)
 if sxx<=0:raise ValueError("experience must increase")
 b=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sxx
 resid=[y-(ym+b*(x-xm)) for x,y in zip(xs,ys)]
 se=math.sqrt(sum(r*r for r in resid)/max(1,len(xs)-2)/sxx)
 return b,b-1.96*se,b+1.96*se
def evaluate_contract(checkpoints):
 a=slope_ci95([(c.E,c.A) for c in checkpoints]);cost=slope_ci95([(c.E,c.C_new) for c in checkpoints])
 live=[c.live_minus_frozen for c in checkpoints[1:]]
 return {"dA_dE":{"mean":a[0],"ci95":[a[1],a[2]]},"dCnew_dE":{"mean":cost[0],"ci95":[cost[1],cost[2]]},
 "live_gt_frozen_fraction":sum(x>0 for x in live)/max(1,len(live)),
 "passed":a[1]>0 and cost[2]<0 and all(x>0 for x in live)}
