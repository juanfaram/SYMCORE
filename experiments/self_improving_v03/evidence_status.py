#!/usr/bin/env python3
"""Evidence semantics: integrity != validity != causality."""
from dataclasses import dataclass
@dataclass(frozen=True)
class EvidenceStatus:
 integrity:bool
 validity:bool
 causality:bool
 def decision(self):
  return "VERIFIED_CAPABILITY" if self.integrity and self.validity and self.causality else "UNVERIFIED"
