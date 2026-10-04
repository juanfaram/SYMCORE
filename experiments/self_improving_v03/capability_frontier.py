#!/usr/bin/env python3
"""Capability frontier: only accept candidates that improve at least one axis without unacceptable regressions."""
from dataclasses import dataclass
@dataclass(frozen=True)
class Capability:
    quality:float; adaptation:float; retention:float; efficiency:float
    def dominates(self,other,tolerance=.03):
        a=(self.quality,self.adaptation,self.retention,self.efficiency);b=(other.quality,other.adaptation,other.retention,other.efficiency)
        no_bad=all(x>=y-tolerance for x,y in zip(a,b))
        better=any(x>y+tolerance for x,y in zip(a,b))
        return no_bad and better
class Frontier:
    def __init__(self):self.items=[]
    def consider(self,name,c):
        if any(old.dominates(c) for _,old in self.items):return False
        self.items=[x for x in self.items if not c.dominates(x[1])]
        self.items.append((name,c));return True
