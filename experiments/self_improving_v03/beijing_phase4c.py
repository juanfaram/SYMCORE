#!/usr/bin/env python3
"""F4C independent Beijing PM2.5 real-host replication with weekly moving-block bootstrap."""
import copy,json,math,random,statistics
from pathlib import Path
from beijing_real_host import BeijingControl,BeijingSymcore,normalize_row
from real_cost_meter import CostMeter,normalized_vector
from BEIJING_PHASE4C_PREREGISTRATION import *
def slope(xs,ys):
 xm=statistics.fmean(xs);ym=statistics.fmean(ys);return sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
def checkpoint_series(obs):
 out=[]
 for cp in CHECKPOINTS:
  upto=[x for x in obs if x[0]<=cp and x[0]>max(EVAL_START,cp-2900)]
  if upto:out.append({"E":cp,"A":statistics.fmean(x[1] for x in upto),"F":statistics.fmean(x[2] for x in upto)})
 return out
def bootstrap_lcb(obs,seed=44031):
 rng=random.Random(seed);eligible=[x for x in obs if x[0]>EVAL_START];blocks=[eligible[i:i+BLOCK] for i in range(0,len(eligible)-BLOCK+1,BLOCK)];vals=[]
 for _ in range(BOOTSTRAPS):
  sample=[]
  while len(sample)<len(eligible):sample.extend(rng.choice(blocks))
  sample=sample[:len(eligible)]
  # preserve original E positions while resampling paired advantage blocks
  synth=[(eligible[i][0],sample[i][1],sample[i][2]) for i in range(len(sample))]
  cs=checkpoint_series(synth)
  if len(cs)>=5:vals.append(slope([x["E"] for x in cs],[x["A"] for x in cs]))
 vals.sort();return [vals[int(.025*(len(vals)-1))],vals[int(.975*(len(vals)-1))]]
def run():
 from ucimlrepo import fetch_ucirepo
 ds=fetch_ucirepo(id=UCI_ID);X=ds.data.features;y=ds.data.targets
 control=BeijingControl();live=BeijingSymcore();frozen=None;cm=CostMeter();lm=CostMeter();obs=[];valid=0;missing=0
 for i in range(len(X)):
  target=y.iloc[i,0]
  if target is None or (isinstance(target,float) and math.isnan(target)):missing+=1;continue
  valid+=1;r=normalize_row(X.iloc[i]);target=float(target)
  ce,_=cm.measure(control.step,r,target,True,updates=1);le,_=lm.measure(live.step,r,target,True,updates=4)
  if valid==FREEZE_AT:frozen=copy.deepcopy(live)
  fe,_=frozen.step(r,target,False) if frozen is not None else (le,0)
  obs.append((valid,ce-le,fe-le))
 checks=checkpoint_series(obs);xs=[x["E"] for x in checks];da=slope(xs,[x["A"] for x in checks]);cia=bootstrap_lcb(obs);rate=sum(x["F"]>0 for x in checks)/len(checks)
 cost=normalized_vector(lm.snapshot())
 out={"schema":"symcore.phase4c.beijing-real-host.v1","dataset":DATASET,"raw_rows":len(X),"valid_interactions":valid,"missing_targets_skipped":missing,
 "external_data":True,"independent_of_prior_hosts":True,"freeze_at":FREEZE_AT,"eval_start":EVAL_START,"checkpoints":checks,
 "dA_dE":{"mean":da,"moving_block_bootstrap_ci95":cia,"block_hours":BLOCK,"B":BOOTSTRAPS},"live_gt_frozen_rate":rate,"cost_final":cost,
 "contract":CONTRACT,"passed":cia[0]>0 and rate>=CONTRACT["live_gt_frozen_rate_gte"]}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/beijing_phase4c_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
