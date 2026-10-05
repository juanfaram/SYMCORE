#!/usr/bin/env python3
"""Causal host-state representation. Every feature is computable before the next intervention decision."""
from collections import deque
import math,statistics
class HostState:
 def __init__(self,window=64):
  self.err=deque(maxlen=window);self.regret=deque(maxlen=window);self.disagree=deque(maxlen=window)
 def observe(self,host_error,reference_error=None,expert_predictions=None):
  self.err.append(float(host_error))
  if reference_error is not None:self.regret.append(float(host_error-reference_error))
  if expert_predictions and len(expert_predictions)>1:self.disagree.append(statistics.pstdev(expert_predictions))
 def vector(self):
  e=list(self.err);half=max(1,len(e)//2)
  level=statistics.fmean(e[-min(16,len(e)):]) if e else 0.
  velocity=(statistics.fmean(e[-half:])-statistics.fmean(e[:half])) if len(e)>=4 else 0.
  regret=statistics.fmean(self.regret) if self.regret else 0.
  disagreement=statistics.fmean(self.disagree) if self.disagree else 0.
  if len(e)>=8:
   recent=statistics.fmean(e[-4:]);past=statistics.fmean(e[:-4]);scale=statistics.pstdev(e) or 1.;change=1/(1+math.exp(-(recent-past)/scale))
  else:change=.5
  return {"error_level":level,"error_velocity":velocity,"recent_regret":regret,"expert_disagreement":disagreement,"change_probability":change}
