#!/usr/bin/env python3
"""Coarse causal risk attribution with Wilson bounds; enough support before specialization."""
import csv,json,math
from collections import defaultdict
from pathlib import Path
from main import ensure
from real_host_control import Host,SymcoreHost
def wilson(k,n,z=1.96):
 p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/d;return max(0,c-h),min(1,c+h)
def bucket(r):
 h=int(r["hr"]);daypart="night" if h<6 else "morning" if h<12 else "afternoon" if h<18 else "evening"
 return (daypart,"work" if r["workingday"]=="1" else "off")
def run(rmax=.05,min_n=200):
 p=Path("data/hour.csv");ensure(p);c=Host(("hr","workingday"));s=SymcoreHost();g=defaultdict(lambda:[0,0,0.])
 with p.open() as f:
  for r in csv.DictReader(f):
   y=float(r["cnt"]);cl=c.step(r,y);sl=s.step(r,y);reg=(sl-cl)/max(abs(cl),1e-9);x=g[bucket(r)];x[0]+=1;x[1]+=reg>rmax;x[2]+=reg
 rows=[]
 for k,(n,bad,total) in g.items():
  if n>=min_n:
   lo,hi=wilson(bad,n);rows.append({"context":{"daypart":k[0],"work":k[1]},"n":n,"violations":bad,"p_hat":bad/n,"p_ci95":[lo,hi],"mean_regression":total/n})
 rows.sort(key=lambda x:x["p_ci95"][1],reverse=True)
 out={"rmax":rmax,"min_n":min_n,"contexts":rows,"worst":rows[:3]}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/coarse_risk_attribution.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
