#!/usr/bin/env python3
"""v1.2 capability recombination: compose validated skills rather than only mutate them."""
from dataclasses import dataclass
@dataclass(frozen=True)
class Skill:
    name:str;domains:frozenset;strengths:dict;cost:float=1.
class Composer:
    def compose(self,a,b,name=None):
        strengths={}
        for k in set(a.strengths)|set(b.strengths):
            strengths[k]=max(a.strengths.get(k,0),b.strengths.get(k,0))
        # modest integration tax makes composition earn its keep
        return Skill(name or f"{a.name}+{b.name}",a.domains|b.domains,strengths,(a.cost+b.cost)*.65)
    def utility(self,s,need):
        coverage=sum(s.strengths.get(k,0)*w for k,w in need.items())
        return coverage/max(.1,s.cost)
