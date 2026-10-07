#!/usr/bin/env python3
"""Finite-state evidence policy: OBSERVE must eventually resolve."""
from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class Observation:
    generation:int
    effect:float
    ci_low:float
    ci_high:float
class ObservePolicy:
    def __init__(self,min_replicates=3,max_generations=5):
        self.min_replicates=min_replicates;self.max_generations=max_generations
    def decide(self,history):
        if not history:return "OBSERVE"
        xs=sorted(history,key=lambda x:x.generation)
        latest=xs[-1]
        if len(xs)>=self.min_replicates and latest.ci_low>0:return "SURVIVE"
        if len(xs)>=self.min_replicates and latest.ci_high<=0:return "REJECT"
        age=latest.generation-xs[0].generation+1
        if age>=self.max_generations:
            return "LIMITED" if latest.effect>0 else "REJECT"
        return "OBSERVE"
