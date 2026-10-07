#!/usr/bin/env python3
"""Evidence-driven research scheduler with protected exploration and anti-exploitation."""
from dataclasses import dataclass
import math,random
@dataclass
class Line:
    name:str;n:int=0;value:float=0.;cost:float=1.;failures:int=0
    def utility(self,total):
        mean=self.value/max(1,self.n);uncertainty=math.sqrt(2*math.log(total+2)/max(1,self.n))
        reliability=1/(1+self.failures/max(1,self.n))
        return (mean+uncertainty)*reliability/max(.1,self.cost)
class ResearchScheduler:
    def __init__(self,names,seed=71,exploit=.70,explore=.20,anti=.10):
        if abs(exploit+explore+anti-1)>1e-9:raise ValueError("budget shares must sum to 1")
        self.lines={n:Line(n) for n in names};self.rng=random.Random(seed)
        self.shares=(exploit,explore,anti)
    def choose(self):
        xs=list(self.lines.values());unseen=[x for x in xs if not x.n]
        r=self.rng.random();exploit,explore,_=self.shares
        if r<explore and unseen:return self.rng.choice(unseen).name
        if r>=explore+exploit:
            # Deliberately revisit structurally neglected/failed lines instead of converging forever.
            return min(xs,key=lambda x:(x.n,-x.failures,self.rng.random())).name
        total=sum(x.n for x in xs)
        return max(xs,key=lambda x:x.utility(total)).name
    def record(self,name,capability_gain,cost=1.,passed=True):
        x=self.lines[name];x.n+=1;x.value+=max(-1.,min(1.,capability_gain));x.cost=.8*x.cost+.2*max(.1,cost);x.failures+=not passed
    def snapshot(self):return {k:vars(v) for k,v in self.lines.items()}
