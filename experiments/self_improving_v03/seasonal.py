#!/usr/bin/env python3
"""Online seasonal memory models: strong causal specialists for recurring interactions."""
from collections import defaultdict,deque
class SeasonalMemory:
    def __init__(self,keys=("hr",),window=8):
        self.keys=keys;self.window=window;self.mem=defaultdict(lambda:deque(maxlen=window));self.global_mem=deque(maxlen=window)
    def _key(self,r):return tuple(r[k] for k in self.keys)
    def predict(self,r):
        xs=self.mem[self._key(r)]
        base=xs if xs else self.global_mem
        return sum(base)/len(base) if base else 0.
    def learn(self,r,y):
        self.mem[self._key(r)].append(float(y));self.global_mem.append(float(y))
