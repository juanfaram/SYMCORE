#!/usr/bin/env python3
"""Risk attribution: identify when/where assisted host causes >Rmax regression."""
import csv,json,statistics
from collections import defaultdict
from pathlib import Path
from main import ensure
from real_host_control import Host,SymcoreHost
def run(rmax=.05):
 p=Path("data/hour.csv");ensure(p);c=Host(("hr","workingday"));s=SymcoreHost();groups=defaultdict(lambda:[0,0,0.])
 with p.open() as f:
  for r in csv.DictReader(f):
   y=float(r["cnt"]);cl=c.step(r,y);sl=s.step(r,y);reg=(sl-cl)/max(abs(cl),1e-9);k=(r["hr"],r["workingday"],r["weathersit"])
   g=groups[k];g[0]+=1;g[1]+=reg>rmax;g[2]+=reg
 rows=[{"context":{"hr":k[0],"workingday":k[1],"weather":k[2]},"n":v[0],"violation_rate":v[1]/v[0],"mean_regression":v[2]/v[0]} for k,v in groups.items() if v[0]>=20]
 rows.sort(key=lambda x:(x["violation_rate"],x["mean_regression"]),reverse=True)
 out={"rmax":rmax,"worst_contexts":rows[:25],"contexts":len(rows)}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/risk_attribution.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
