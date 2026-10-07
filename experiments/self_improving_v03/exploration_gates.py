#!/usr/bin/env python3
"""Separate exploratory existence from validated promotion."""
from dataclasses import dataclass
@dataclass(frozen=True)
class CandidateEvidence:
    harmful:bool
    novelty:float
    redundancy:float
    quality:float=0.;adaptation:float=0.;retention:float=0.;efficiency:float=0.
def existence_gate(e,novelty_floor=.20,redundancy_ceiling=.90):
    return (not e.harmful) and e.novelty>=novelty_floor and e.redundancy<=redundancy_ceiling
def promotion_gate(e,floors=None):
    floors=floors or {"quality":.0,"adaptation":.0,"retention":.0,"efficiency":.0}
    return existence_gate(e) and all(getattr(e,k)>=v for k,v in floors.items())
