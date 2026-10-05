#!/usr/bin/env python3
"""Learned bidirectional Pi: contextual operation value from evolutionary experience."""
from __future__ import annotations
import math,random
OPS=("create","modify","combine","freeze","prune","restore","noop")
class LearnedEvolutionPi:
 def __init__(self,kernel_sigma=.24,prior=0.0,temperature=.10):
  self.sigma=float(kernel_sigma);self.prior=float(prior);self.temperature=float(temperature);self.memory=[]
 def observe(self,z,op,value):
  if op not in OPS:raise ValueError(op)
  self.memory.append((tuple(float(x) for x in z),op,float(value)))
 def _weight(self,a,b):
  d2=sum((x-y)**2 for x,y in zip(a,b));return math.exp(-d2/(2*self.sigma*self.sigma))
 def value(self,z,op):
  rows=[(self._weight(z,x),v) for x,o,v in self.memory if o==op]
  den=1.;num=self.prior
  for w,v in rows:num+=w*v;den+=w
  return num/den
 def probabilities(self,z):
  vals={o:self.value(z,o) for o in OPS};mx=max(vals.values())
  ex={o:math.exp((v-mx)/self.temperature) for o,v in vals.items()};s=sum(ex.values())
  return {o:v/s for o,v in ex.items()}
 def choose(self,z,rng):
  ps=self.probabilities(z);u=rng.random();acc=0.
  for o in OPS:
   acc+=ps[o]
   if u<=acc:return o
  return OPS[-1]
 def frozen(self):
  q=LearnedEvolutionPi(self.sigma,self.prior,self.temperature);q.memory=list(self.memory);return q
