#!/usr/bin/env python3
"""Collect contextual V evidence on opened hosts and compare contextual vs context-free Pi_learn LOHO."""
import csv,json,math
from pathlib import Path
from main import ensure
from seasonal import SeasonalMemory
from metro_real_host import MetroSymcore,normalize_row as metro_norm
from beijing_real_host import BeijingSymcore,normalize_row as bj_norm
from vlearn_context_collector import collect
from pi_learn import PiLearn,features
from pi_learn_value import score
from PI_LEARN_CONTEXT_CONTRACT import *
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
 p=Path("data/hour.csv");ensure(p)
 with p.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def datasets():
 from ucimlrepo import fetch_ucirepo
 out={}
 out["bike"]=collect(bike_rows(),BikeLearner,lambda r:r,lambda r:float(r["cnt"]))
 m=fetch_ucirepo(id=492);mr=[dict(m.data.features.iloc[i],__target=float(m.data.targets.iloc[i,0])) for i in range(len(m.data.features))]
 out["metro"]=collect(mr,MetroSymcore,metro_norm,lambda r:r["__target"])
 b=fetch_ucirepo(id=381);br=[dict(b.data.features.iloc[i],__target=b.data.targets.iloc[i,0]) for i in range(len(b.data.features))]
 out["beijing"]=collect(br,BeijingSymcore,bj_norm,lambda r:r["__target"])
 return out
def eval_policy(train,test,contextual):
 p=PiLearn(k=min(12,max(3,len(train))),margin=.5)
 for r in train:p.observe(features(r["E"],r["h"],r["z"] if contextual else ()),r["V"])
 dec=[]
 for r in test:
  a,_=p.decide(features(r["E"],r["h"],r["z"] if contextual else ()));dec.append((a,r["V"]))
 return score(dec)
def run():
 ds=datasets();out={};vi=0;rr=0
 for held in HOSTS:
  train=[r for h,rows in ds.items() if h!=held for r in rows];test=ds[held]
  base=eval_policy(train,test,False);ctx=eval_policy(train,test,True)
  dv=ctx["value_capture"]-base["value_capture"];dr=base["mean_regret"]-ctx["mean_regret"]
  vi+=dv>0;rr+=dr>0;out[held]={"n":len(test),"baseline":base,"contextual":ctx,"delta_value_capture":dv,"regret_reduction":dr}
 passed=vi>=CONTRACT["value_capture_improved_hosts_gte"] and rr>=CONTRACT["mean_regret_reduced_hosts_gte"]
 report={"schema":"symcore.pi-learn-context-loho.v1","status":STATUS,"contract":CONTRACT,"hosts":out,
 "value_capture_improved_hosts":vi,"mean_regret_reduced_hosts":rr,"passed":passed}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/pi_learn_context_loho.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return report
if __name__=="__main__":run()
