#!/usr/bin/env python3
"""Metro real-host adapter: causal seasonal memories over external chronological observations."""
from __future__ import annotations
import math,statistics
from collections import defaultdict,deque
class MetroMemory:
 def __init__(self,keys,window=8):self.keys=tuple(keys);self.window=window;self.mem=defaultdict(lambda:deque(maxlen=window));self.global_mem=deque(maxlen=window)
 def key(self,r):return tuple(r[k] for k in self.keys)
 def predict(self,r):
  xs=self.mem[self.key(r)];base=xs if xs else self.global_mem
  return statistics.fmean(base) if base else 0.
 def learn(self,r,y):self.mem[self.key(r)].append(float(y));self.global_mem.append(float(y))
class MetroControl:
 def __init__(self):self.m=MetroMemory(("hour","weekday"),8)
 def step(self,r,y,learn=True):
  p=self.m.predict(r)
  if learn:self.m.learn(r,y)
  return abs(p-y),int(learn)
class MetroSymcore:
 def __init__(self):
  self.experts={"hour":MetroMemory(("hour",),8),"weekday":MetroMemory(("hour","weekday"),8),
   "weather":MetroMemory(("hour","weather_main"),8),"holiday":MetroMemory(("hour","holiday"),8)}
  self.loss={k:deque(maxlen=256) for k in self.experts}
 def step(self,r,y,learn=True):
  preds={k:m.predict(r) for k,m in self.experts.items()}
  mature={k:statistics.fmean(v) for k,v in self.loss.items() if len(v)>=64};pick=min(mature,key=mature.get) if mature else "hour"
  err=abs(preds[pick]-y)
  if learn:
   for k,m in self.experts.items():self.loss[k].append(abs(preds[k]-y));m.learn(r,y)
  return err,(len(self.experts) if learn else 0)
def normalize_row(row):
 from datetime import datetime
 dt=datetime.strptime(str(row["date_time"]),"%Y-%m-%d %H:%M:%S")
 return {"hour":str(dt.hour),"weekday":str(dt.weekday()),"weather_main":str(row["weather_main"]),"holiday":str(row["holiday"])}
