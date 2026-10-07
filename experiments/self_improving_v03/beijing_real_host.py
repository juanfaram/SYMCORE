#!/usr/bin/env python3
"""Causal online adapter for external Beijing PM2.5 stream."""
import statistics
from collections import defaultdict,deque
class Memory:
 def __init__(self,keys,w=8):self.keys=tuple(keys);self.mem=defaultdict(lambda:deque(maxlen=w));self.global_mem=deque(maxlen=w)
 def key(self,r):return tuple(r[k] for k in self.keys)
 def predict(self,r):
  x=self.mem[self.key(r)];b=x if x else self.global_mem
  return statistics.fmean(b) if b else 0.
 def learn(self,r,y):self.mem[self.key(r)].append(float(y));self.global_mem.append(float(y))
class BeijingControl:
 def __init__(self):self.m=Memory(("hour","month"),8)
 def step(self,r,y,learn=True):
  p=self.m.predict(r)
  if learn:self.m.learn(r,y)
  return abs(p-y),int(learn)
class BeijingSymcore:
 def __init__(self):
  self.experts={"hour":Memory(("hour",),8),"season":Memory(("hour","month"),8),
                "wind":Memory(("hour","cbwd"),8),"weather":Memory(("hour","temp_bin"),8)}
  self.loss={k:deque(maxlen=256) for k in self.experts}
 def step(self,r,y,learn=True):
  ps={k:m.predict(r) for k,m in self.experts.items()};mature={k:statistics.fmean(v) for k,v in self.loss.items() if len(v)>=64};pick=min(mature,key=mature.get) if mature else "hour";err=abs(ps[pick]-y)
  if learn:
   for k,m in self.experts.items():self.loss[k].append(abs(ps[k]-y));m.learn(r,y)
  return err,(len(self.experts) if learn else 0)
def normalize_row(row):
 temp=float(row["TEMP"])
 return {"hour":str(int(row["hour"])),"month":str(int(row["month"])),"cbwd":str(row["cbwd"]),"temp_bin":str(int(temp//5))}
