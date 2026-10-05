#!/usr/bin/env python3
"""Transferable causal effect memory: gap -> intervention effect -> verified outcome."""
from collections import defaultdict
from intervention_effects import cosine
class EffectMemory:
    def __init__(self,prior=1.0):self.prior=prior;self.rows=defaultdict(list)
    def observe(self,gap,effect,outcome):self.rows[gap].append((effect,float(outcome)))
    def value(self,gap,candidate):
        rows=self.rows.get(gap,())
        num=self.prior*.5;den=self.prior
        for eff,y in rows:
            w=max(0.,cosine(eff,candidate))**2
            num+=w*y;den+=w
        return num/den
    def rank(self,gap,local_effects):
        return sorted(local_effects,key=lambda k:self.value(gap,local_effects[k]),reverse=True)
