#!/usr/bin/env python3
"""Pi_learn: uncertainty-aware policy over LEARN/FREEZE/MEASURE from V_learn evidence."""
from __future__ import annotations
import math,statistics
ACTIONS=("LEARN","FREEZE","MEASURE")
class PiLearn:
 def __init__(self,k=12,margin=.5):self.k=k;self.margin=margin;self.rows=[]
 def observe(self,features,value):self.rows.append((tuple(map(float,features)),float(value)))
 def _dist(self,a,b):
  # log scales make E and h comparable; z components are expected normalized.
  return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
 def predict(self,features):
  x=tuple(map(float,features));near=sorted(((self._dist(x,z),v) for z,v in self.rows),key=lambda q:q[0])[:self.k]
  if not near:return {"mean":0.,"halfwidth":float("inf"),"support":0}
  ws=[1/(1e-6+d) for d,_ in near];sw=sum(ws);m=sum(w*v for w,(_,v) in zip(ws,near))/sw
  var=sum(w*(v-m)**2 for w,(_,v) in zip(ws,near))/sw;neff=sw*sw/sum(w*w for w in ws)
  hw=1.96*math.sqrt(var/max(1.,neff))
  return {"mean":m,"halfwidth":hw,"support":len(near)}
 def decide(self,features):
  p=self.predict(features);lo=p["mean"]-p["halfwidth"];hi=p["mean"]+p["halfwidth"]
  if lo>self.margin:return "LEARN",p
  if hi<-self.margin:return "FREEZE",p
  return "MEASURE",p
def features(E,h,z=()):
 return (math.log1p(E),math.log1p(h),*tuple(map(float,z)))
