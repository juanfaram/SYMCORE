#!/usr/bin/env python3
"""Universal functional state geometry from anonymous causal sensor traces."""
from __future__ import annotations
import math,statistics
from collections import deque
class FunctionalGeometry:
 def __init__(self,window=64):self.window=window;self.channels={}
 def observe(self,signals):
  # Signal names are local adapter metadata only; geometry never exposes them.
  for k,v in signals.items():
   x=float(v)
   if not math.isfinite(x):continue
   q=self.channels.setdefault(k,deque(maxlen=self.window));q.append(x)
 def _signature(self,x):
  if not x:return (0.,0.,0.,0.,0.)
  n=len(x);m=statistics.fmean(x);sd=statistics.pstdev(x) or 1e-9
  norm=[(v-m)/sd for v in x]
  half=max(1,n//2);trend=statistics.fmean(norm[-half:])-statistics.fmean(norm[:half]) if n>=4 else 0.
  volatility=statistics.pstdev(norm) if n>1 else 0.
  if n>=3:
   a=norm[:-1];b=norm[1:];am=statistics.fmean(a);bm=statistics.fmean(b)
   den=math.sqrt(sum((v-am)**2 for v in a)*sum((v-bm)**2 for v in b));memory=sum((u-am)*(v-bm) for u,v in zip(a,b))/den if den else 0.
  else:memory=0.
  recent=statistics.fmean(norm[-min(8,n):])
  shock=max(abs(v) for v in norm[-min(8,n):])
  return (trend,volatility,memory,recent,shock)
 def vector(self):
  # Sort signatures by functional shape, not channel name: invariant to renaming/reordering.
  sigs=sorted(self._signature(list(q)) for q in self.channels.values() if len(q)>=4)
  if not sigs:return (0.,)*15
  cols=list(zip(*sigs))
  # Distribution of channel dynamics: mean, dispersion, extreme for each relational statistic.
  out=[]
  for col in cols:
   out.extend((statistics.fmean(col),statistics.pstdev(col) if len(col)>1 else 0.,max(abs(x) for x in col)))
  return tuple(out)
def distance(a,b):
 if len(a)!=len(b):raise ValueError("dimension mismatch")
 return math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/max(1,len(a)))
