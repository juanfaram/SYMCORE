#!/usr/bin/env python3
import csv,json
from pathlib import Path
from seasonal import SeasonalMemory
from guarded_router import GuardedRouter
def run():
 e={"hour":SeasonalMemory(("hr",),8),"hour_work":SeasonalMemory(("hr","workingday"),8),"hour_day":SeasonalMemory(("hr","weekday"),6),"hour_weather":SeasonalMemory(("hr","weathersit"),6)}
 r=GuardedRouter(e,"hour_day",256,.02);errs=[];choices={}
 with Path("data/hour.csv").open() as f:
  for row in csv.DictReader(f):
   y=float(row["cnt"]);ctx="work" if row["workingday"]=="1" else "off";p,c=r.learn(row,y,ctx);errs.append(abs(p-y));choices[c]=choices.get(c,0)+1
 out={"guarded_mae_last512":round(sum(errs[-512:])/512,3),"choices":choices}
 Path("artifacts/guarded_router_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":
 if not Path("data/hour.csv").exists():
  from main import ensure;ensure(Path("data/hour.csv"))
 run()
