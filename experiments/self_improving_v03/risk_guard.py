#!/usr/bin/env python3
"""Evidence-derived risk guard: abstain in coarse contexts whose prior upper risk bound exceeds budget."""
class RiskGuard:
 def __init__(self,unsafe_contexts):self.unsafe={tuple(sorted(x.items())) for x in unsafe_contexts}
 def allow(self,context):return tuple(sorted(context.items())) not in self.unsafe
