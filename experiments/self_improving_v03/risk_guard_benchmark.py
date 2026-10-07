#!/usr/bin/env python3
"""Evaluate risk guard on a chronological holdout: attribution train half, guard test half."""
import csv,json,math
from collections import defaultdict
from pathlib import Path
from main import ensure
from real_host_control import Host,SymcoreHost
from risk_contract import risk_report,passes
from coarse_risk_attribution import bucket,wilson
def run(rmax=.05,delta=.05):
 p=Path("data/hour.csv");ensure(p);rows=list(csv.DictReader(p.open()));half=len(rows)//2
 c=Host(("hr","workingday"));s=SymcoreHost();g=defaultdict(lambda:[0,0])
 # train attribution causally on first half
 for r in rows[:half]:
  y=float(r["cnt"]);cl=c.step(r,y);sl=s.step(r,y);reg=(sl-cl)/max(abs(cl),1e-9);x=g[bucket(r)];x[0]+=1;x[1]+=reg>rmax
 unsafe=set()
 for k,(n,bad) in g.items():
  if n>=100 and wilson(bad,n)[1]>=delta:unsafe.add(k)
 # fresh models on holdout; fallback is online control in unsafe contexts
 c=Host(("hr","workingday"));s=SymcoreHost();cl=[];gl=[];active=0
 for r in rows[half:]:
  y=float(r["cnt"]);ce=c.step(r,y);se=s.step(r,y);allow=bucket(r) not in unsafe;cl.append(ce);gl.append(se if allow else ce);active+=allow
 rep=risk_report(cl,gl,rmax);gain=sum(a-b for a,b in zip(cl,gl))/len(cl)
 out={"unsafe_contexts":[list(x) for x in sorted(unsafe)],"active_fraction":active/len(cl),"mean_gain":gain,"risk":rep,"risk_pass":passes(rep,delta),"delta":delta}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/risk_guard_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
