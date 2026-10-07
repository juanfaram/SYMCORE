#!/usr/bin/env python3
import csv,json
from pathlib import Path
from seasonal import SeasonalMemory
from residual_specialist import ResidualSpecialist
def run():
 anchor=SeasonalMemory(("hr","weekday"),6);m=ResidualSpecialist(anchor,keys=("hr","workingday"),window=16);err=[]
 with Path("data/hour.csv").open() as f:
  for r in csv.DictReader(f):
   y=float(r["cnt"]);p=m.predict(r)
   if anchor.global_mem:err.append(abs(p-y))
   m.learn(r,y)
 out={"residual_mae_last512":round(sum(err[-512:])/512,3)}
 Path("artifacts/residual_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":
 if not Path("data/hour.csv").exists():
  from main import ensure;ensure(Path("data/hour.csv"))
 run()
