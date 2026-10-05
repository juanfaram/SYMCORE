#!/usr/bin/env python3
"""Cross-host contract: SYMCORE must improve multiple structurally different online hosts."""
import json,math,random
from pathlib import Path
from host_advantage import paired_advantage
def stream(seed,n,kind):
 r=random.Random(seed)
 for t in range(n):
  if kind=="periodic":y=50+25*math.sin(2*math.pi*t/24)+r.gauss(0,4)
  elif kind=="regime":y=(20 if t<n//2 else 80)+r.gauss(0,6)
  else:y=.03*t+15*math.sin(2*math.pi*t/50)+r.gauss(0,5)
  yield t,y
def trial(seed,kind,n=3000):
 last=0.;season={};resid={};cl=[];sl=[]
 for t,y in stream(seed,n,kind):
  c=last
  key=t%24 if kind=="periodic" else (0 if kind=="regime" else t%50)
  base=season.get(key,last);corr=resid.get(key,0.);s=base+corr
  if t>100:cl.append(abs(c-y));sl.append(abs(s-y))
  # same causal observations, different adaptation policy
  last=y;old=season.get(key,y);season[key]=.8*old+.2*y;resid[key]=.85*corr+.15*(y-base)
 return cl,sl
def run():
 out={}
 for kind in ("periodic","regime","trend"):
  C=[];S=[]
  for seed in (2,7,19,43,89):
   c,s=trial(seed,kind);C+=c;S+=s
  out[kind]=paired_advantage(C,S)
 out["universal_pass"]=sum(v["passed"] for v in out.values())>=2
 Path("artifacts/cross_host_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
