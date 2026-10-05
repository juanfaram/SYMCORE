#!/usr/bin/env python3
"""Risk envelope on the same paired real stream used for host advantage."""
import csv,json
from pathlib import Path
from main import ensure
from real_host_control import Host,SymcoreHost
from risk_contract import risk_report,passes
def run():
 p=Path("data/hour.csv");ensure(p);c=Host(("hr","workingday"));s=SymcoreHost();cl=[];sl=[]
 with p.open() as f:
  for r in csv.DictReader(f):
   y=float(r["cnt"]);cl.append(c.step(r,y));sl.append(s.step(r,y))
 rep=risk_report(cl[512:],sl[512:],.05);rep["passed_delta_05"]=passes(rep,.05)
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/real_risk_report.json").write_text(json.dumps(rep,indent=2));print(json.dumps(rep,indent=2));return rep
if __name__=="__main__":run()
