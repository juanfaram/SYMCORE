#!/usr/bin/env python3
"""Necessity Destruction for functional state: prune dimensions only when causal cross-host value survives."""
from __future__ import annotations
from itertools import combinations
def project(z,keep):return tuple(z[i] for i in keep)
def aggregate_host_scores(results):
 # results: host -> {value_capture, mean_regret}
 return {"mean_value_capture":sum(x["value_capture"] for x in results.values())/len(results),
         "mean_regret":sum(x["mean_regret"] for x in results.values())/len(results)}
def necessity_destroy(full_dims,evaluate,tol_value=.002,tol_regret=.02):
 """
 evaluate(keep_dims)-> host score dict.
 Greedy destruction: delete the dimension whose removal causes least damage.
 A deletion is legal only if mean value capture drops <=tol_value and regret rises <=tol_regret.
 Re-run until no legal deletion remains. Returns complete audit trail.
 """
 keep=tuple(full_dims);trail=[];base=evaluate(keep);b=aggregate_host_scores(base)
 while len(keep)>1:
  candidates=[]
  for d in keep:
   k=tuple(x for x in keep if x!=d);r=evaluate(k);a=aggregate_host_scores(r)
   dv=a["mean_value_capture"]-b["mean_value_capture"];dr=a["mean_regret"]-b["mean_regret"]
   candidates.append((dv-dr,k,d,r,a,dv,dr))
  candidates.sort(key=lambda x:x[0],reverse=True)
  _,k,d,r,a,dv,dr=candidates[0]
  legal=dv>=-tol_value and dr<=tol_regret
  trail.append({"removed":d,"candidate_keep":k,"delta_value_capture":dv,"delta_mean_regret":dr,"legal":legal})
  if not legal:break
  keep=k;base=r;b=a
 return {"survivors":keep,"scores":base,"aggregate":b,"trail":trail,"stop":True}
