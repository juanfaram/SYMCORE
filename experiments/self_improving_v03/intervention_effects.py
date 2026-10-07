#!/usr/bin/env python3
"""Domain-independent intervention effects: the transferable language between Pi and hosts."""
from dataclasses import dataclass
@dataclass(frozen=True)
class Effect:
    plasticity:float=0.;stability:float=0.;reset:float=0.;representation:float=0.;exploration:float=0.
    def vector(self):return (self.plasticity,self.stability,self.reset,self.representation,self.exploration)
def cosine(a,b):
 av=a.vector();bv=b.vector();dot=sum(x*y for x,y in zip(av,bv));na=sum(x*x for x in av)**.5;nb=sum(x*x for x in bv)**.5
 return dot/max(1e-12,na*nb)
# Universal intent prototypes; adapters map local mechanisms onto these effects.
INTENTS={
 "adapt_fast":Effect(plasticity=1.,exploration=.7,stability=-.3),
 "consolidate":Effect(stability=1.,plasticity=-.2),
 "reset_state":Effect(reset=1.,plasticity=.6),
 "expand_representation":Effect(representation=1.,exploration=.3),
 "balanced_adapt":Effect(plasticity=.55,stability=.45,exploration=.25)}
def choose_local(intent,local_effects):
 target=INTENTS[intent]
 return max(local_effects,key=lambda name:cosine(target,local_effects[name]))
