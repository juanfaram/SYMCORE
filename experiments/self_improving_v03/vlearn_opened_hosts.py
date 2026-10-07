#!/usr/bin/env python3
"""Run V_learn surface on three already-opened external hosts. Exploratory, not new holdout."""
import json
from pathlib import Path
from vlearn_surface import evaluate_surface,summarize
from seasonal import SeasonalMemory
from metro_real_host import MetroSymcore,normalize_row as metro_norm
from beijing_real_host import BeijingSymcore,normalize_row as bj_norm
from main import ensure
class BikeLearner:
 def __init__(self):
  from collections import deque
  self.experts={"hour":SeasonalMemory(("hr",),8),"work":SeasonalMemory(("hr","workingday"),8),"weekday":SeasonalMemory(("hr","weekday"),8),"weather":SeasonalMemory(("hr","weathersit"),8)}
  self.loss={k:deque(maxlen=256) for k in self.experts}
 def step(self,r,y,learn=True):
  import statistics
  ps={k:m.predict(r) for k,m in self.experts.items()};mature={k:statistics.fmean(v) for k,v in self.loss.items() if len(v)>=64};pick=min(mature,key=mature.get) if mature else "hour";err=abs(ps[pick]-y)
  if learn:
   for k,m in self.experts.items():self.loss[k].append(abs(ps[k]-y));m.learn(r,y)
  return err,(len(self.experts) if learn else 0)
def bike_rows():
 import csv
 p=Path("data/hour.csv");ensure(p)
 with p.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def run():
 from ucimlrepo import fetch_ucirepo
 bike=evaluate_surface(bike_rows(),BikeLearner,lambda r:r,lambda r:float(r["cnt"]))
 metro=fetch_ucirepo(id=492);mr=[dict(metro.data.features.iloc[i],__target=float(metro.data.targets.iloc[i,0])) for i in range(len(metro.data.features))]
 ms=evaluate_surface(mr,MetroSymcore,metro_norm,lambda r:r["__target"])
 bj=fetch_ucirepo(id=381);br=[dict(bj.data.features.iloc[i],__target=bj.data.targets.iloc[i,0]) for i in range(len(bj.data.features)]
 bs=evaluate_surface(br,BeijingSymcore,bj_norm,lambda r:r["__target"])
 surfaces={"bike":bike,"metro":ms,"beijing":bs};summary={k:summarize(v) for k,v in surfaces.items()}
 out={"schema":"symcore.vlearn.opened-hosts.v1","confirmatory":False,"analysis_status":"posthoc_on_opened_hosts","surfaces":surfaces,"summary":summary,"cl2_descriptive_all_hosts_pass":all(x["passed"] for x in summary.values())}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/vlearn_opened_hosts_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
