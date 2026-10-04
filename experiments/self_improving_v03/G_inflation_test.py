#!/usr/bin/env python3
"""Strict G: count only expansion against immutable accumulated frontier, reject duplicates."""
import json
from pathlib import Path
from capability_frontier import Capability
def dominates(a,b,t=.03):
 av=a.values();bv=b.values();return all(x>=y-t for x,y in zip(av,bv)) and any(x>y+t for x,y in zip(av,bv))
def run():
 stream=[("a",Capability(.8,.8,.8,.8)),("trivial",Capability(.79,.8,.8,.8)),("b",Capability(.84,.84,.84,.84)),("duplicate",Capability(.84,.84,.84,.84))]
 archive=[];strict=[]
 for n,c in stream:
  duplicate=any(all(abs(x-y)<1e-12 for x,y in zip(c.values(),old.values())) for _,old in archive)
  dominated=any(dominates(old,c) for _,old in archive)
  expands=not duplicate and not dominated and (not archive or any(dominates(c,old) for _,old in archive))
  if expands:strict.append(n)
  archive.append((n,c))
 out={"naive_G_count":len(stream),"strict_G_count":len(strict),"inflation_ratio":len(stream)/max(1,len(strict)),"strict_capabilities":strict}
 Path("artifacts/G_inflation_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
