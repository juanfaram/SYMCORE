#!/usr/bin/env python3
"""v1.6 reusable latent representation learned from tasks, independent of action identities."""
from collections import defaultdict
class StructuralRepresentation:
    """Learns which context factors are predictive across skills, then transfers factor relevance."""
    def __init__(self):
        self.factor_signal=defaultdict(lambda:[0,0.0])
    def observe(self,context,reward):
        for factor in ("domain","difficulty","task"):
            key=(factor,context.get(factor));s=self.factor_signal[key];s[0]+=1;s[1]+=abs(float(reward))
    def relevance(self,context):
        out={}
        for factor in ("domain","difficulty","task"):
            n,r=self.factor_signal[(factor,context.get(factor))];out[factor]=r/n if n else 0.
        z=sum(out.values()) or 1.;return {k:v/z for k,v in out.items()}
    def keys(self,context):
        rel=self.relevance(context)
        return sorted(((k,context.get(k),w) for k,w in rel.items()),key=lambda x:x[2],reverse=True)
