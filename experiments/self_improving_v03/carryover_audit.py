#!/usr/bin/env python3
"""Carryover audit around frozen selective intervention transitions."""
import csv,json,statistics
from pathlib import Path
from main import ensure
from real_host_control import Host,SymcoreHost
from intervention_controller import InterventionController
def run(k=10):
 p=Path("data/hour.csv");ensure(p);rows=list(csv.DictReader(p.open()));o=Host(("hr","workingday"));s=SymcoreHost();ctl=InterventionController();d=[];g=[]
 for r in rows:
  y=float(r["cnt"]);a=ctl.decide();oe=o.step(r,y);se=s.step(r,y);d.append(oe-se);g.append(int(a));ctl.observe(oe)
 starts=[i for i in range(1,len(g)) if g[i] and not g[i-1] and i>=k and i+k<len(g)]
 pre=[statistics.fmean(d[i-k:i]) for i in starts];during=[d[i] for i in starts];post=[statistics.fmean(d[i+1:i+k+1]) for i in starts]
 out={"transitions":len(starts),"window":k,"pre_mean":statistics.fmean(pre) if pre else 0,"onset_mean":statistics.fmean(during) if during else 0,
      "post_mean":statistics.fmean(post) if post else 0,"carryover":(statistics.fmean(post)-statistics.fmean(pre)) if pre else 0}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/carryover_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
