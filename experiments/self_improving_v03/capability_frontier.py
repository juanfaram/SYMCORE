#!/usr/bin/env python3
"""Robust evolutionary capability frontier: no scalar utility, uncertainty-aware Pareto evidence."""
from dataclasses import dataclass
AXES=("quality","adaptation","retention","efficiency","learnability")
@dataclass(frozen=True)
class Interval:
 mean:float;low:float;high:float
@dataclass(frozen=True)
class Capability:
 quality:Interval;adaptation:Interval;retention:Interval;efficiency:Interval;learnability:Interval
 def values(self):return tuple(getattr(self,a) for a in AXES)
 def robustly_dominates(self,other,tolerances=None):
  tol=tolerances or {a:0.0 for a in AXES};a=self.values();b=other.values()
  # Non-inferiority is judged conservatively from confidence bounds.
  noninferior=all(x.low >= y.low-tol[k] for k,(x,y) in zip(AXES,zip(a,b)))
  # At least one axis must have separated confidence intervals: evidence of genuine expansion.
  strict=any(x.low > y.high+tol[k] for k,(x,y) in zip(AXES,zip(a,b)))
  return noninferior and strict
class Frontier:
 def __init__(self,regression_budgets=None):
  self.items=[];self.regression_budgets=regression_budgets or {a:float("inf") for a in AXES}
 def intolerable_regression(self,candidate,baseline):
  # Compare to the actual incumbent/baseline, never an impossible vector of per-axis maxima.
  return any(c.low < b.low-self.regression_budgets[a] for a,c,b in zip(AXES,candidate.values(),baseline.values()))
 def consider(self,name,c,baseline=None):
  if baseline is not None and self.intolerable_regression(c,baseline):return False
  if any(old.robustly_dominates(c) for _,old in self.items):return False
  self.items=[x for x in self.items if not c.robustly_dominates(x[1])]
  self.items.append((name,c));return True
