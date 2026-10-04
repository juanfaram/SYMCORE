#!/usr/bin/env python3
"""SYMCORE v0.8 autonomous research scheduler: allocate finite experiment budget from evidence."""
from dataclasses import dataclass,field
import math,random,json
from pathlib import Path
@dataclass
class Line:
    name:str;n:int=0;value:float=0.;cost:float=1.;failures:int=0
    def utility(self,total):
        mean=self.value/max(1,self.n)
        uncertainty=math.sqrt(2*math.log(total+2)/max(1,self.n))
        reliability=1/(1+self.failures/max(1,self.n))
        return (mean+uncertainty)*reliability/max(.1,self.cost)
class ResearchScheduler:
    def __init__(self,names,seed=71):self.lines={n:Line(n) for n in names};self.rng=random.Random(seed)
    def choose(self):
        unseen=[x for x in self.lines.values() if not x.n]
        if unseen:return self.rng.choice(unseen).name
        total=sum(x.n for x in self.lines.values())
        return max(self.lines.values(),key=lambda x:x.utility(total)).name
    def record(self,name,capability_gain,cost=1.,passed=True):
        x=self.lines[name];x.n+=1;x.value+=max(-1.,min(1.,capability_gain));x.cost=.8*x.cost+.2*max(.1,cost);x.failures+=not passed
    def snapshot(self):return {k:vars(v) for k,v in self.lines.items()}
