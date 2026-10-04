#!/usr/bin/env python3
"""Validated capability frontier with protected-axis regression floors."""
from dataclasses import dataclass
@dataclass(frozen=True)
class Capability:
    quality:float;adaptation:float;retention:float;efficiency:float
    def values(self):return (self.quality,self.adaptation,self.retention,self.efficiency)
    def dominates(self,other,tolerances=(.03,.03,.03,.03)):
        a=self.values();b=other.values()
        return all(x>=y-t for x,y,t in zip(a,b,tolerances)) and any(x>y+t for x,y,t in zip(a,b,tolerances))
class Frontier:
    def __init__(self,protected_floor_drop=.05):self.items=[];self.floor_drop=protected_floor_drop
    def consider(self,name,c):
        # Universal safety invariant: no candidate may regress > floor on ANY axis
        # relative to every validated capability merely because it occupies a new niche.
        if self.items:
            reference=tuple(max(old.values()[i] for _,old in self.items) for i in range(4))
            if any(v<r-self.floor_drop for v,r in zip(c.values(),reference)):return False
        if any(old.dominates(c) for _,old in self.items):return False
        self.items=[x for x in self.items if not c.dominates(x[1])]
        self.items.append((name,c));return True
