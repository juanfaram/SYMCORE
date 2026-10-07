#!/usr/bin/env python3
"""Causal evolutionary memory: predicts Frontier deltas and cost from functional state + effect."""
from __future__ import annotations
import math,statistics
from dataclasses import dataclass
AXES=("quality","adaptation","retention","efficiency","learnability")
@dataclass(frozen=True)
class Prediction:
 means:dict;ci95:dict;cost_mean:float;cost_ci95:float;support:int
class CausalEvolutionMemory:
 def __init__(self,k=64):self.k=k;self.rows=[]
 def observe(self,z,e,delta,cost):
  if len(delta)!=len(AXES):raise ValueError("five frontier deltas required")
  self.rows.append((tuple(map(float,z)),tuple(map(float,e)),tuple(map(float,delta)),float(cost)))
 def _dist(self,z,e,row):
  rz,re,_,_=row
  dz=sum((a-b)**2 for a,b in zip(z,rz))/max(1,len(z));de=sum((a-b)**2 for a,b in zip(e,re))/max(1,len(e))
  return math.sqrt(dz+de)
 def predict(self,z,e):
  near=sorted(self.rows,key=lambda r:self._dist(z,e,r))[:self.k]
  if not near:raise ValueError("empty memory")
  ds=[self._dist(z,e,r) for r in near];ws=[1/(1e-5+d) for d in ds];sw=sum(ws)
  means={};cis={}
  for j,a in enumerate(AXES):
   m=sum(w*r[2][j] for w,r in zip(ws,near))/sw
   var=sum(w*(r[2][j]-m)**2 for w,r in zip(ws,near))/sw
   neff=sw*sw/sum(w*w for w in ws);ci=1.96*math.sqrt(var/max(1.,neff))
   means[a]=m;cis[a]=ci
  cm=sum(w*r[3] for w,r in zip(ws,near))/sw;cv=sum(w*(r[3]-cm)**2 for w,r in zip(ws,near))/sw;neff=sw*sw/sum(w*w for w in ws)
  return Prediction(means,cis,cm,1.96*math.sqrt(cv/max(1.,neff)),len(near))
 def choose(self,z,effects,risk_budget=.05):
  preds={n:self.predict(z,e) for n,e in effects.items()}
  # Pareto-style predicted utility: admissible effects cannot have confidently bad axes; then maximize robust positive expansion per cost.
  def score(p):
   lows=[p.means[a]-p.ci95[a] for a in AXES]
   if min(lows)<-risk_budget:return -1e9
   return sum(max(0.,x) for x in lows)/max(1e-6,p.cost_mean+p.cost_ci95)
  return max(preds,key=lambda n:score(preds[n])),preds
