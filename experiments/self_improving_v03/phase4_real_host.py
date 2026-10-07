#!/usr/bin/env python3
"""Phase 4: three-arm real-data longitudinal test on external UCI Bike Sharing stream."""
from __future__ import annotations
import csv,json,math,statistics,random
from copy import deepcopy
from pathlib import Path
from main import ensure
from seasonal import SeasonalMemory
from PHASE4_PREREGISTRATION import *
class Control:
 def __init__(self):self.m=SeasonalMemory(("hr","workingday"),8)
 def step(self,r,y,learn=True):
  p=self.m.predict(r)
  if learn:self.m.learn(r,y)
  return abs(p-y),1 if learn else 0
class SymcoreOnline:
 def __init__(self):
  self.experts={"hour":SeasonalMemory(("hr",),8),"work":SeasonalMemory(("hr","workingday"),8),
   "weekday":SeasonalMemory(("hr","weekday"),8),"weather":SeasonalMemory(("hr","weathersit"),8)}
  self.loss={k:[] for k in self.experts}
 def step(self,r,y,learn=True):
  preds={k:m.predict(r) for k,m in self.experts.items()}
  mature={k:statistics.fmean(v[-256:]) for k,v in self.loss.items() if len(v)>=64}
  chosen=min(mature,key=mature.get) if mature else "hour";err=abs(preds[chosen]-y)
  if learn:
   for k,m in self.experts.items():self.loss[k].append(abs(preds[k]-y));m.learn(r,y)
  return err,(len(self.experts) if learn else 0)
def slope(xs,ys):
 xm=statistics.fmean(xs);ym=statistics.fmean(ys);return sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
def block_bootstrap_slope(rows,key,seed=44004,B=3000):
 rng=random.Random(seed);blocks=[rows[i:i+BLOCK//1024 or 1] for i in range(len(rows))]
 vals=[]
 for _ in range(B):
  sample=[]
  while len(sample)<len(rows):sample.extend(rng.choice(blocks))
  sample=sample[:len(rows)];sample=sorted(sample,key=lambda x:x["E"])
  # duplicates in E can make denominator zero only in pathological resamples
  try:vals.append(slope([x["E"] for x in sample],[x[key] for x in sample]))
  except ZeroDivisionError:pass
 vals.sort();return [vals[int(.025*(len(vals)-1))],vals[int(.975*(len(vals)-1))]]
def run():
 path=Path("data/hour.csv");ensure(path)
 control=Control();live=SymcoreOnline();frozen=None
 ec=[];el=[];ef=[];live_updates=0;check=[]
 with path.open(newline="",encoding="utf-8") as f:
  for t,r in enumerate(csv.DictReader(f),1):
   y=float(r["cnt"])
   ce,_=control.step(r,y,True);le,u=live.step(r,y,True);live_updates+=u
   if t==FREEZE_AT:frozen=deepcopy(live)
   fe,_=frozen.step(r,y,False) if frozen is not None else (le,0)
   ec.append(ce);el.append(le);ef.append(fe)
   if t in CHECKPOINTS:
    lo=max(EVAL_START,t-1024);ix0=lo-1
    c=statistics.fmean(ec[ix0:t]);l=statistics.fmean(el[ix0:t]);fr=statistics.fmean(ef[ix0:t])
    # Higher performance = negative MAE. A and F therefore positive when live has lower error.
    A=c-l;F=fr-l
    # Empirical acquisition cost per unit current advantage; numerator is actual incremental specialist updates.
    cumulative_updates=max(1,live_updates);Cnew=cumulative_updates/max(1e-6,max(A,1e-6))/t
    check.append({"E":t,"A":A,"F":F,"C_new":Cnew,"control_mae":c,"live_mae":l,"frozen_mae":fr})
 if len(check)<MIN_CHECKPOINTS:raise RuntimeError("insufficient checkpoints")
 xs=[x["E"] for x in check];da=slope(xs,[x["A"] for x in check]);dc=slope(xs,[x["C_new"] for x in check])
 cia=block_bootstrap_slope(check,"A");cic=block_bootstrap_slope(check,"C_new",44005)
 rate=sum(x["F"]>0 for x in check)/len(check)
 out={"schema":"symcore.phase4.real-host.v1","dataset":DATASET,"rows":len(ec),"external_data":True,
 "freeze_at":FREEZE_AT,"eval_start":EVAL_START,"checkpoints":check,
 "dA_dE":{"mean":da,"ci95":cia},"dCnew_dE":{"mean":dc,"ci95":cic},"live_gt_frozen_rate":rate,
 "contract":CONTRACT,"passed":cia[0]>0 and cic[1]<0 and rate>=CONTRACT["live_gt_frozen_rate_gte"]}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/phase4_real_host_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
