#!/usr/bin/env python3
"""Compare naive ledger growth G with strict frontier-expansion G."""
import json
from pathlib import Path
from capability_frontier import Capability,Frontier
def run():
 candidates=[("a",Capability(.8,.8,.8,.8)),("trivial",Capability(.79,.8,.8,.8)),("b",Capability(.84,.81,.81,.81)),("duplicate",Capability(.84,.81,.81,.81))]
 naive=len(candidates);f=Frontier(.05);accepted=[]
 for n,c in candidates:
  before=set(x for x,_ in f.items)
  if f.consider(n,c):
   after=set(x for x,_ in f.items)
   # strict growth counts only actual expansion, not merely a ledger event
   if n in after and (not before or after!=before):accepted.append(n)
 strict=len(set(accepted));out={"naive_G_count":naive,"strict_G_count":strict,"inflation_ratio":round(naive/max(1,strict),3),"strict_capabilities":accepted}
 Path("artifacts/G_inflation_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
