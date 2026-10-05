#!/usr/bin/env python3
"""Operator policy Π(op|gap): learn which intervention families produce verified growth."""
from __future__ import annotations
from collections import defaultdict
import math,random
OPERATORS=("parameters","features","memory","representation","structure","objectives","learning")
class OperatorPolicy:
    def __init__(self,operators=OPERATORS,prior=1.0,history_strength=1.0):
        self.operators=tuple(operators);self.prior=float(prior);self.history_strength=float(history_strength)
        self.success=defaultdict(lambda:defaultdict(float));self.trials=defaultdict(lambda:defaultdict(float))
    def update(self,gap_type,operator,survived,weight=1.0):
        if operator not in self.operators:raise ValueError(operator)
        self.trials[gap_type][operator]+=weight
        self.success[gap_type][operator]+=weight*float(bool(survived))
    def distribution(self,gap_type):
        # Beta-Bernoulli posterior mean, normalized. Never collapses exploration to zero.
        raw={op:(self.success[gap_type][op]+self.prior)/(self.trials[gap_type][op]+2*self.prior) for op in self.operators}
        z=sum(raw.values());return {op:v/z for op,v in raw.items()}
    def choose(self,gap_type,rng=None):
        rng=rng or random;d=self.distribution(gap_type);x=rng.random();acc=0.
        for op,p in d.items():
            acc+=p
            if x<=acc:return op
        return self.operators[-1]
    def entropy(self,gap_type):
        return -sum(p*math.log(max(p,1e-12)) for p in self.distribution(gap_type).values())
    def ingest_verified_history(self,rows,weight=None):
        # Only externally validated decisions teach Π. OBSERVE/REJECT do not become positive evidence.
        for r in rows:
            gap=r.get("gap_type") or r.get("weakness") or "unknown"
            op=r.get("operator")
            if op in self.operators:self.update(gap,op,r.get("decision")=="SURVIVE",self.history_strength if weight is None else weight)
        return self
