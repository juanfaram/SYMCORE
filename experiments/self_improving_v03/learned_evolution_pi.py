#!/usr/bin/env python3
"""Experience-learned Pi with operation-specific contextual geometry and uncertainty."""
from __future__ import annotations
import math
OPS=("create","modify","combine","freeze","prune","restore","noop")
class LearnedEvolutionPi:
 def __init__(self,k=48,temperature=.035,exploration=.04):
  self.k=int(k);self.temperature=float(temperature);self.exploration=float(exploration);self.memory=[];self.scales={}
 def observe(self,z,op,value):
  self.memory.append((tuple(map(float,z)),op,float(value)))
 def fit(self):
  # Learn which state coordinates predict each operation's value from experience itself.
  for op in OPS:
   rows=[(z,v) for z,o,v in self.memory if o==op]
   if not rows:self.scales[op]=(1.,)*7;continue
   ys=[v for _,v in rows];ym=sum(ys)/len(ys);weights=[]
   for j in range(len(rows[0][0])):
    xs=[z[j] for z,_ in rows];xm=sum(xs)/len(xs)
    cov=sum((x-xm)*(y-ym) for x,y in zip(xs,ys));vx=sum((x-xm)**2 for x in xs);vy=sum((y-ym)**2 for y in ys)
    corr=abs(cov/math.sqrt(max(1e-12,vx*vy)))
    weights.append(.15+corr)
   self.scales[op]=tuple(weights)
  return self
 def _dist(self,z,x,op):
  w=self.scales.get(op,(1.,)*len(z));return sum(a*(u-v)**2 for a,u,v in zip(w,z,x))
 def value(self,z,op):
  rows=sorted(((self._dist(z,x,op),v) for x,o,v in self.memory if o==op),key=lambda q:q[0])[:self.k]
  if not rows:return 0.
  # Local inverse-distance regression preserves specialized regions instead of global smoothing.
  ws=[1./(1e-5+d) for d,_ in rows];return sum(w*v for w,(_,v) in zip(ws,rows))/sum(ws)
 def probabilities(self,z):
  vals={o:self.value(z,o) for o in OPS};mx=max(vals.values());ex={o:math.exp((v-mx)/self.temperature) for o,v in vals.items()};s=sum(ex.values())
  base={o:v/s for o,v in ex.items()};eps=self.exploration
  return {o:(1-eps)*base[o]+eps/len(OPS) for o in OPS}
 def choose(self,z,rng):
  ps=self.probabilities(z);u=rng.random();acc=0.
  for o in OPS:
   acc+=ps[o]
   if u<=acc:return o
  return OPS[-1]
 def frozen(self):
  q=LearnedEvolutionPi(self.k,self.temperature,self.exploration);q.memory=list(self.memory);q.scales=dict(self.scales);return q
