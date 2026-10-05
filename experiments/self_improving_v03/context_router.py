#!/usr/bin/env python3
"""Contextual tabular router: learn when each already-validated specialist wins; specialists remain frozen in definition."""
import math
from collections import defaultdict
class ContextRouter:
 def __init__(self,n_experts,epsilon=.05):
  self.n=n_experts;self.epsilon=epsilon;self.stats=defaultdict(lambda:[[0,0.0] for _ in range(n_experts)]);self.t=0
 def key(self,x):
  # coarse online-observable bins; no label leakage
  return tuple(int(min(2,max(0,v*3))) for v in x[:3])
 def choose(self,x):
  self.t+=1;k=self.key(x);s=self.stats[k]
  unseen=[i for i,(n,r) in enumerate(s) if n==0]
  if unseen:return unseen[0]
  return max(range(self.n),key=lambda i:s[i][1]/s[i][0]+math.sqrt(2*math.log(self.t+1)/s[i][0]))
 def feedback(self,x,losses):
  k=self.key(x)
  for i,l in enumerate(losses):
   s=self.stats[k][i];s[0]+=1;s[1]+=1-float(l)
