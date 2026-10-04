#!/usr/bin/env python3
"""Coevolutionary examiner: generate harder regimes from observed learner weaknesses."""
import random
class Examiner:
    def __init__(self,seed=404):self.rng=random.Random(seed);self.weakness={}
    def observe(self,context,correct):
        k=(context["domain"],context["difficulty"]);s=self.weakness.setdefault(k,[0,0]);s[0]+=not correct;s[1]+=1
    def next_context(self):
        domains=("code","analysis","writing","forecast","rare");diffs=("simple","normal","hard")
        candidates=[(d,q) for d in domains for q in diffs]
        def hardness(x):
            bad,n=self.weakness.get(x,[0,0]);rate=bad/n if n else .5
            rarity=1/(n+1)
            return rate+0.35*rarity+self.rng.random()*.05
        d,q=max(candidates,key=hardness)
        return {"task":"solve","domain":d,"difficulty":q}
