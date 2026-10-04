#!/usr/bin/env python3
"""v1.3 residual specialist: strong seasonal anchor + online residual correction."""
from collections import defaultdict,deque
class ResidualSpecialist:
    def __init__(self,anchor,alpha=.12,keys=("hr","workingday"),window=32):
        self.anchor=anchor;self.alpha=alpha;self.keys=keys;self.resid=defaultdict(lambda:deque(maxlen=window))
    def key(self,r):return tuple(r[k] for k in self.keys)
    def predict(self,r):
        base=self.anchor.predict(r);xs=self.resid[self.key(r)]
        return max(0.,base+(sum(xs)/len(xs) if xs else 0.))
    def learn(self,r,y):
        base=self.anchor.predict(r);self.resid[self.key(r)].append(float(y)-base);self.anchor.learn(r,y)
