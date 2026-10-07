#!/usr/bin/env python3
"""Direct acquisition-cost instrumentation. Measures resources; never divides by advantage."""
from __future__ import annotations
import time,tracemalloc
from dataclasses import dataclass,asdict
@dataclass(frozen=True)
class CostObservation:
 interactions:int
 updates:int
 wall_seconds:float
 peak_bytes:int
@dataclass
class CostMeter:
 interactions:int=0
 updates:int=0
 wall_seconds:float=0.
 peak_bytes:int=0
 def measure(self,fn,*args,updates=1,**kwargs):
  # Per-interaction direct measurement; no outcome/advantage enters the cost definition.
  tracemalloc.start()
  t0=time.perf_counter()
  try:return fn(*args,**kwargs)
  finally:
   dt=time.perf_counter()-t0
   _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
   self.interactions+=1;self.updates+=int(updates);self.wall_seconds+=dt;self.peak_bytes=max(self.peak_bytes,int(peak))
 def snapshot(self):
  return CostObservation(self.interactions,self.updates,self.wall_seconds,self.peak_bytes)
def delta(a:CostObservation,b:CostObservation):
 return CostObservation(b.interactions-a.interactions,b.updates-a.updates,b.wall_seconds-a.wall_seconds,max(0,b.peak_bytes-a.peak_bytes))
def normalized_vector(x:CostObservation):
 # Report dimensions independently. No arbitrary scalarization.
 return {"seconds_per_interaction":x.wall_seconds/max(1,x.interactions),
         "updates_per_interaction":x.updates/max(1,x.interactions),
         "peak_bytes":x.peak_bytes,
         "interactions":x.interactions}
