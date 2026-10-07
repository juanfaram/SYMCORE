#!/usr/bin/env python3
"""Run Necessity Destruction on full 15-D functional geometry using paired LOHO causal value."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
from main import ensure
from seasonal import SeasonalMemory
from metro_real_host import MetroSymcore,normalize_row as metro_norm
from beijing_real_host import BeijingSymcore,normalize_row as bj_norm
from vlearn_full_geometry_collector import collect_full_geometry
from pi_learn import PiLearn,features
from pi_learn_value import score
from necessity_destroy_geometry import necessity_destroy
from GEOMETRY_PRUNING_CONTRACT import *
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
def numeric_signals(r):
 out={}
 for k,v in r.items():
  try:out[k]=float(v)
  except (TypeError,ValueError):pass
 return out
def bike_rows():
 p=Path("data/hour.csv");ensure(p)
 with p.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def build_data():
 from ucimlrepo import fetch_ucirepo
 out={}
 out["bike"]=collect_full_geometry(bike_rows(),BikeLearner,lambda r:r,lambda r:float(r["cnt"]),numeric_signals)
 m=fetch_ucirepo(id=492);mr=[dict(m.data.features.iloc[i],__target=float(m.data.targets.iloc[i,0])) for i in range(len(m.data.features))]
 out["metro"]=collect_full_geometry(mr,MetroSymcore,metro_norm,lambda r:r["__target"],numeric_signals)
 b=fetch_ucirepo(id=381);br=[dict(b.data.features.iloc[i],__target=b.data.targets.iloc[i,0]) for i in range(len(b.data.features))]
 out["beijing"]=collect_full_geometry(br,BeijingSymcore,bj_norm,lambda r:r["__target"],numeric_signals)
 return out
def host_eval(data,keep):
 result={}
 for held in data:
  train=[r for h,rows in data.items() if h!=held for r in rows];test=data[held]
  p=PiLearn(k=min(12,max(3,len(train))),margin=.5)
  for r in train:p.observe(features(r["E"],r["h"],tuple(r["z"][i] for i in keep)),r["V"])
  dec=[]
  for r in test:
   a,_=p.decide(features(r["E"],r["h"],tuple(r["z"][i] for i in keep)));dec.append((a,r["V"]))
  result[held]=score(dec)
 return result
def run():
 data=build_data()
 full=host_eval(data,FULL_DIMS)
 pruned=necessity_destroy(FULL_DIMS,lambda keep:host_eval(data,keep),MAX_VALUE_CAPTURE_DROP,MAX_MEAN_REGRET_RISE)
 empty=host_eval(data,())
 # Improvement vs E,h baseline is evaluated host-by-host, not only in aggregate.
 improved=sum(pruned["scores"][h]["value_capture"]>empty[h]["value_capture"] for h in data)
 report={"schema":"symcore.geometry-necessity-destruction.v1","status":STATUS,"full_dims":FULL_DIMS,
 "baseline_Eh":empty,"full_geometry":full,"pruned":pruned,"improved_vs_Eh_hosts":improved,
 "passes_transfer_gate":improved>=MIN_HOSTS_IMPROVED_VS_EH}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/geometry_necessity_destruction.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return report
if __name__=="__main__":run()
