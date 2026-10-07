#!/usr/bin/env python3
"""Blind evidence gate: candidate origin/lineage is deliberately unavailable to evaluator."""
from dataclasses import dataclass
@dataclass(frozen=True)
class BlindEvidence:
 quality:float;adaptation:float;retention:float;efficiency:float;robustness:float;samples:int
def evaluate(e,min_samples=5):
 if e.samples<min_samples:return False
 protected=(e.quality,e.adaptation,e.retention,e.robustness)
 return min(protected)>=.6 and e.efficiency>=.4
