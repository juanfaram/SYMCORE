#!/usr/bin/env python3
"""Phase-1 instrument validation: negative/positive controls and multi-seed stability."""
import json,math,random,statistics
from pathlib import Path
from online_continuous_learning import Checkpoint,evaluate_contract
CHECKPOINTS=(100,200,300,400,500,600)
def series(seed,learning):
 r=random.Random(seed);rows=[]
 for t in CHECKPOINTS:
  # Shared noise cancels in paired causal advantage; residual noise tests the estimator.
  noise=r.gauss(0,.0015)
  A=(.00035*t if learning else 0.)+noise
  C=(1-.0007*t if learning else 1.)+r.gauss(0,.001)
  live_frozen=(.00030*(t-100) if learning and t>100 else 0.)+r.gauss(0,.0008)
  rows.append(Checkpoint(t,A,C,live_frozen))
 return rows
def slope(xs,ys):
 xm=statistics.fmean(xs);ym=statistics.fmean(ys);return sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
def run():
 seeds=range(20);pos=[];neg=[]
 for s in seeds:
  p=series(81000+s,True);n=series(91000+s,False)
  pos.append(slope([x.E for x in p],[x.A for x in p]));neg.append(slope([x.E for x in n],[x.A for x in n]))
 pm=statistics.fmean(pos);ps=statistics.stdev(pos);cv=ps/abs(pm)
 nm=statistics.fmean(neg);nse=statistics.stdev(neg)/math.sqrt(len(neg));plo=pm-1.96*ps/math.sqrt(len(pos))
 out={"schema":"symcore.online-instrument.phase1.v1","seeds":len(pos),"positive_slope_mean":pm,"positive_slope_lcb95":plo,
      "negative_slope_mean":nm,"negative_slope_ci95":[nm-1.96*nse,nm+1.96*nse],"positive_cv":cv,
      "criteria":{"positive_lcb_gt_zero":True,"negative_ci_contains_zero":True,"max_cv":.20},
      "passed":plo>0 and (nm-1.96*nse)<=0<=(nm+1.96*nse) and cv<.20}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/online_phase1_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
