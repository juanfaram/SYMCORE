#!/usr/bin/env python3
"""Executable invariant north-star gate; unknown evidence remains UNKNOWN, never silently PASS."""
from dataclasses import dataclass
@dataclass
class NorthStar:
 delta_omega:bool|None=None
 A:bool|None=None
 dA_dE:bool|None=None
 risk:bool|None=None
 def status(self):
  vals={"delta_omega":self.delta_omega,"A":self.A,"dA_dE":self.dA_dE,"risk":self.risk}
  return {"conditions":vals,"all_proven":all(v is True for v in vals.values()),"unknown":[k for k,v in vals.items() if v is None],"failed":[k for k,v in vals.items() if v is False]}
