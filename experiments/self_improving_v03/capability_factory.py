#!/usr/bin/env python3
"""Capability factory: create bounded new specialists from detected gaps and related skills."""
from dataclasses import dataclass
@dataclass
class SpecialistSpec:
    name:str;context:dict;parents:list;memory:int;exploration:float
class CapabilityFactory:
    def propose(self,gap,registry):
        ctx=gap["context"]
        related=sorted(registry,key=lambda r:sum(r.get("context",{}).get(k)==v for k,v in ctx.items()),reverse=True)[:2]
        parents=[r["name"] for r in related]
        slug="-".join(str(ctx.get(k,"x")) for k in ("domain","difficulty"))
        return SpecialistSpec("skill-"+slug,ctx,parents,256,.12 if gap["best"]<.4 else .06)
