#!/usr/bin/env python3
"""Counterfactual evolution Pi: learns operation advantage/regret relative to alternatives."""
from __future__ import annotations
import math
OPS=("create","modify","combine","freeze","prune","restore","noop")
class CounterfactualEvolutionPi:
 def __init__(self,k=36,temperature=.025,exploration=.025):
  self.k=k;self.temperature=temperature;self.exploration=exploration;self.rows=[];self.metric=(1.,)*7
 def observe_counterfactual(self,z,values):
  best=max(values.values());noop=values["noop"]
  # Store the full action-value vector for the same state: paired counterfactual evidence.
  self.rows.append((tuple(map(float,z)),{o:(values[o]-noop,best-values[o]) for o in OPS}))
 def fit(self):
  # Learn a shared state metric from variation in pairwise action advantages.
  ys=[max(v[o][0] for o in OPS)-min(v[o][0] for o in OPS) for _,v in self.rows];ym=sum(ys)/len(ys)
  w=[]
  for j in range(7):
   xs=[z[j] for z,_ in self.rows];xm=sum(xs)/len(xs);cov=sum((x-xm)*(y-ym) for x,y in zip(xs,ys));vx=sum((x-xm)**2 for x in xs);vy=sum((y-ym)**2 for y in ys)
   w.append(.2+abs(cov/math.sqrt(max(1e-12,vx*vy))))
  self.metric=tuple(w);return self
 def _d(self,a,b):return sum(w*(x-y)**2 for w,x,y in zip(self.metric,a,b))
 def scores(self,z):
  near=sorted(((self._d(z,x),v) for x,v in self.rows),key=lambda q:q[0])[:self.k]
  ws=[1/(1e-6+d) for d,_ in near];den=sum(ws)
  out={}
  for o in OPS:
   advantage=sum(w*v[o][0] for w,(_,v) in zip(ws,near))/den
   regret=sum(w*v[o][1] for w,(_,v) in zip(ws,near))/den
   out[o]=advantage-regret
  return out
 def probabilities(self,z):
  s=self.scores(z);mx=max(s.values());ex={o:math.exp((v-mx)/self.temperature) for o,v in s.items()};den=sum(ex.values());eps=self.exploration
  return {o:(1-eps)*ex[o]/den+eps/len(OPS) for o in OPS}
 def choose(self,z,rng):
  ps=self.probabilities(z);u=rng.random();a=0.
  for o in OPS:
   a+=ps[o]
   if u<=a:return o
  return OPS[-1]
 def frozen(self):
  q=CounterfactualEvolutionPi(self.k,self.temperature,self.exploration);q.rows=[(z,dict(v)) for z,v in self.rows];q.metric=self.metric;return q
