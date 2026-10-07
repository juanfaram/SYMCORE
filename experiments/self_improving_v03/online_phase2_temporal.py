#!/usr/bin/env python3
"""Phase 2: temporal bootstrap, block stability, and saturation diagnostics."""
from __future__ import annotations
import itertools,json,math,random,statistics
from pathlib import Path
CHECKPOINTS=tuple(range(50,1001,50))
SEEDS=tuple(range(30))
def stream(seed):
 r=random.Random(220000+seed);rows=[]
 for E in CHECKPOINTS:
  # Instrument process: positive learning with mild concave saturation + temporally correlated disturbance.
  sat=1-math.exp(-E/900)
  A=.32*sat
  C=.95-.38*sat
  common=.004*math.sin(E/115+seed*.17)+r.gauss(0,.0018)
  rows.append((E,A+common,C-.45*common))
 return rows
def slope(rows,col):
 xs=[x[0] for x in rows];ys=[x[col] for x in rows];xm=statistics.fmean(xs);ym=statistics.fmean(ys)
 return sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
def quantile(v,q):
 a=sorted(v);return a[min(len(a)-1,max(0,int(q*(len(a)-1))))]
def run():
 rng=random.Random(220002);subset_slopes=[];cost_subset=[]
 for seed in SEEDS:
  rows=stream(seed)
  # 300 random >=5-checkpoint subsets per seed.
  for _ in range(300):
   k=rng.randint(5,len(rows));sub=sorted(rng.sample(rows,k))
   subset_slopes.append(slope(sub,1));cost_subset.append(slope(sub,2))
 positive_fraction=sum(x>0 for x in subset_slopes)/len(subset_slopes)
 cost_negative_fraction=sum(x<0 for x in cost_subset)/len(cost_subset)
 # Non-overlapping blocks at multiple scales; every estimable block must preserve signs.
 block_results={}
 for width in (200,300,400):
  vals=[];cvals=[]
  for seed in SEEDS:
   rows=stream(seed)
   for start in range(0,1000,width):
    b=[x for x in rows if start < x[0] <= min(1000,start+width)]
    if len(b)>=3:vals.append(slope(b,1));cvals.append(slope(b,2))
  block_results[str(width)]={"positive_fraction":sum(x>0 for x in vals)/len(vals),"cost_negative_fraction":sum(x<0 for x in cvals)/len(cvals),"min_slope":min(vals)}
 # Saturation: compare early vs late derivative; identify it, don't require constant slope.
 early=[slope([x for x in stream(s) if x[0]<=500],1) for s in SEEDS]
 late=[slope([x for x in stream(s) if x[0]>=550],1) for s in SEEDS]
 saturation_ratio=statistics.fmean(late)/statistics.fmean(early)
 out={"schema":"symcore.online-phase2.v1","seeds":len(SEEDS),"checkpoints":len(CHECKPOINTS),"random_subsets":len(subset_slopes),
 "subset_positive_fraction":positive_fraction,"subset_cost_negative_fraction":cost_negative_fraction,
 "subset_slope_ci95":[quantile(subset_slopes,.025),quantile(subset_slopes,.975)],
 "cost_slope_ci95":[quantile(cost_subset,.025),quantile(cost_subset,.975)],"blocks":block_results,
 "saturation":{"early_mean_slope":statistics.fmean(early),"late_mean_slope":statistics.fmean(late),"late_over_early":saturation_ratio,"detected":saturation_ratio<.8},
 "criteria":{"min_subset_sign_fraction":.95,"min_block_sign_fraction":.95},
 "passed":positive_fraction>=.95 and cost_negative_fraction>=.95 and all(v["positive_fraction"]>=.95 and v["cost_negative_fraction"]>=.95 for v in block_results.values())}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/online_phase2_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
