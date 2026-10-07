#!/usr/bin/env python3
"""v1.1 guarded router: ensemble must earn the right to override a strong specialist."""
from collections import defaultdict,deque
class GuardedRouter:
    def __init__(self,experts,anchor,window=256,min_gain=.02):
        self.experts=experts;self.anchor=anchor;self.window=window;self.min_gain=min_gain
        self.loss=defaultdict(lambda:deque(maxlen=window))
    def predict(self,row,context="global"):
        preds={n:e.predict(row) for n,e in self.experts.items()}
        anchor_loss=self.loss[(context,self.anchor)]
        eligible=[]
        for n in self.experts:
            xs=self.loss[(context,n)]
            if len(xs)>=64 and len(anchor_loss)>=64 and sum(xs)/len(xs)<(sum(anchor_loss)/len(anchor_loss))*(1-self.min_gain):
                eligible.append(n)
        chosen=min(eligible,key=lambda n:sum(self.loss[(context,n)])/len(self.loss[(context,n)])) if eligible else self.anchor
        return preds[chosen],chosen,preds
    def learn(self,row,y,context="global"):
        p,chosen,preds=self.predict(row,context)
        for n,e in self.experts.items():self.loss[(context,n)].append(abs(preds[n]-y));e.learn(row,y)
        return p,chosen
