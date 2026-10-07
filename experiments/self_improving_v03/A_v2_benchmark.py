#!/usr/bin/env python3
"""Re-measure A with calibrated acquisition-cost v2; no motor changes."""
import itertools,json,math,random,statistics
from pathlib import Path
from acquisition_metric import acquisition_cost
def curve(seed,kind,n=600):
 r=random.Random(seed);mem={};loss=[];period={"easy":24,"medium":17,"hard":11,"shift":19}[kind];noise={"easy":2,"medium":7,"hard":14,"shift":9}[kind]
 for i in range(1,n+1):
  base=50+20*math.sin(2*math.pi*i/period)+(20 if kind=="shift" and i>n//2 else 0);y=base+r.gauss(0,noise);k=i%period;p=mem.get(k,50);loss.append(abs(p-y));mem[k]=.8*mem.get(k,y)+.2*y
 return loss
def run():
 Path("artifacts").mkdir(parents=True,exist_ok=True)
 kinds=("easy","medium","hard","shift");seeds=(5,17,41,73,109,163,223);rows=[]
 for order in itertools.permutations(kinds):
  costs=[]
  for s in seeds:costs.append([acquisition_cost(curve(s+j*100,k))["cost"] for j,k in enumerate(order)])
  # This is an instrument-only remeasurement: task difficulty is fixed, so acceleration claim requires order-robust decline after difficulty normalization.
  norm=[]
  base={k:statistics.fmean(acquisition_cost(curve(s,k))["cost"] for s in seeds) for k in kinds}
  for cs in costs:norm.append([c/base[k] for c,k in zip(cs,order)])
  A=[(x[0]-x[-1])/x[0] for x in norm];rows.append({"order":order,"A_mean":statistics.fmean(A),"A_positive_fraction":sum(v>0 for v in A)/len(A)})
 out={"orders":rows,"robust_positive_orders":sum(x["A_positive_fraction"]>.5 for x in rows),"total_orders":len(rows)}
 Path("artifacts/A_v2_report.json").write_text(json.dumps(out,indent=2));print(json.dumps({"robust_positive_orders":out["robust_positive_orders"],"total_orders":len(rows)},indent=2));return out
if __name__=="__main__":run()
