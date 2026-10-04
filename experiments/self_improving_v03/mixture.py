from __future__ import annotations
import math
from collections import defaultdict,deque
class AdaptiveMixture:
    def __init__(self,experts,eta=.08,memory=1024):
        self.experts=experts;self.eta=eta;self.loss=defaultdict(lambda:deque(maxlen=memory));self.context_loss=defaultdict(lambda:defaultdict(lambda:deque(maxlen=memory)))
    def weights(self,context="global"):
        s={}
        for n in self.experts:
            xs=self.context_loss[context][n] or self.loss[n];mean=sum(xs)/len(xs) if xs else 0.;s[n]=math.exp(-self.eta*min(mean,500))
        z=sum(s.values()) or 1.;return {k:v/z for k,v in s.items()}
    def predict(self,row,context="global"):
        w=self.weights(context);p={n:e.predict(row) for n,e in self.experts.items()};return sum(w[n]*p[n] for n in p),p,w
    def learn(self,row,y,context="global"):
        pred,p,w=self.predict(row,context)
        for n,e in self.experts.items():
            loss=abs(p[n]-y);self.loss[n].append(loss);self.context_loss[context][n].append(loss);e.learn(row,y)
        return pred,p,w
class ReplayMemory:
    def __init__(self,capacity=2048):self.capacity=capacity;self.items=[];self.seen=0
    def add(self,item,rng):
        self.seen+=1
        if len(self.items)<self.capacity:self.items.append(item)
        else:
            j=rng.randrange(self.seen)
            if j<self.capacity:self.items[j]=item
    def sample(self,n,rng):return rng.sample(self.items,min(n,len(self.items)))
