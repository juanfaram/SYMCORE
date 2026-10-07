#!/usr/bin/env python3
"""Capability factory: create bounded specialists with an explicit success contract."""
from dataclasses import dataclass
@dataclass
class SpecialistSpec:
    name:str;context:dict;parents:list;memory:int;exploration:float
    success_action:str|None=None
    success_threshold:float=.80
class CapabilityFactory:
    def propose(self,gap,registry,success_action=None):
        ctx=gap["context"]
        related=sorted(registry,key=lambda r:sum(r.get("context",{}).get(k)==v for k,v in ctx.items()),reverse=True)[:2]
        parents=[r["name"] for r in related]
        slug="-".join(str(ctx.get(k,"x")) for k in ("domain","difficulty"))
        # The factory carries what success means; acquisition no longer has to infer its objective from context alone.
        return SpecialistSpec("skill-"+slug,ctx,parents,256,.12 if gap["best"]<.4 else .06,success_action,.80)
