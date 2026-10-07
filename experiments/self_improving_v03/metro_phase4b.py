#!/usr/bin/env python3
"""Independent Phase-4B real-host replication on UCI Metro Interstate Traffic Volume."""
import json,statistics,copy
from pathlib import Path
from metro_real_host import MetroControl,MetroSymcore,normalize_row
from real_cost_meter import CostMeter,normalized_vector
from METRO_PHASE4B_PREREGISTRATION import *
def slope(xs,ys):
 xm=statistics.fmean(xs);ym=statistics.fmean(ys);return sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
def ci_seedless_checkpoint(rows,key):
 # Conservative leave-one-checkpoint-out sensitivity interval; not pseudo-seeds.
 vals=[]
 for j in range(len(rows)):
  s=[x for i,x in enumerate(rows) if i!=j]
  vals.append(slope([x["E"] for x in s],[x[key] for x in s]))
 return [min(vals),max(vals)]
def run():
 from ucimlrepo import fetch_ucirepo
 ds=fetch_ucirepo(id=UCI_ID);X=ds.data.features;y=ds.data.targets
 control=MetroControl();live=MetroSymcore();frozen=None;cm=CostMeter();lm=CostMeter()
 ec=[];el=[];ef=[];checks=[];cost_prev=None
 for i in range(len(X)):
  t=i+1;r=normalize_row(X.iloc[i]);target=float(y.iloc[i,0] if hasattr(y,"iloc") else y[i])
  ce,_=cm.measure(control.step,r,target,True,updates=1)
  le,_=lm.measure(live.step,r,target,True,updates=4)
  if t==FREEZE_AT:frozen=copy.deepcopy(live)
  fe,_=frozen.step(r,target,False) if frozen is not None else (le,0)
  ec.append(ce);el.append(le);ef.append(fe)
  if t in CHECKPOINTS:
   lo=max(EVAL_START,t-3200)-1;c=statistics.fmean(ec[lo:t]);l=statistics.fmean(el[lo:t]);fr=statistics.fmean(ef[lo:t])
   cv=normalized_vector(lm.snapshot());checks.append({"E":t,"A":c-l,"F":fr-l,"control_mae":c,"live_mae":l,"frozen_mae":fr,"cost":cv})
 xs=[x["E"] for x in checks];da=slope(xs,[x["A"] for x in checks]);cia=ci_seedless_checkpoint(checks,"A");rate=sum(x["F"]>0 for x in checks)/len(checks)
 cost_slopes={k:slope(xs,[x["cost"][k] for x in checks]) for k in COST_AXES}
 out={"schema":"symcore.phase4b.metro-real-host.v1","dataset":DATASET,"rows":len(X),"external_data":True,"independent_of_bike":True,
 "freeze_at":FREEZE_AT,"eval_start":EVAL_START,"checkpoints":checks,"dA_dE":{"mean":da,"sensitivity_interval":cia},
 "live_gt_frozen_rate":rate,"cost_vector_slopes":cost_slopes,"contract":CONTRACT,
 "passed":cia[0]>0 and rate>=CONTRACT["live_gt_frozen_rate_gte"]}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/metro_phase4b_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
