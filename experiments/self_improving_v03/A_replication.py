#!/usr/bin/env python3
"""Independent replication of calibrated A: new seeds + second task family, frozen acquisition metric."""
import itertools,json,math,random,statistics
from pathlib import Path
from acquisition_metric import acquisition_cost
FAMILIES={
 "periodic":{
  "p24":(24,3,0),"p16":(16,6,0),"p10":(10,10,0),"shift":(20,7,18)},
 "trend":{
  "slow":(28,4,.015),"medium":(21,7,.035),"fast":(14,10,.065),"shifttrend":(18,8,.045)}
}
SEEDS=(271,313,359,401,457,503,557,601,653,709,761)
def curve(seed,spec,n=700):
 period,noise,slope=spec;r=random.Random(seed);mem={};loss=[]
 for i in range(1,n+1):
  y=45+slope*i+18*math.sin(2*math.pi*i/period)+r.gauss(0,noise);k=i%period;p=mem.get(k,45+slope*i);loss.append(abs(p-y));mem[k]=.8*mem.get(k,y)+.2*y
 return loss
def family(name,tasks):
 base={k:statistics.fmean(acquisition_cost(curve(s,v))["cost"] for s in SEEDS) for k,v in tasks.items()};rows=[]
 for oi,order in enumerate(itertools.permutations(tasks)):
  vals=[]
  for s in SEEDS:
   cs=[acquisition_cost(curve(s+oi*37+j*101,tasks[k]))["cost"]/base[k] for j,k in enumerate(order)]
   vals.append((cs[0]-cs[-1])/cs[0])
  rows.append(statistics.fmean(vals))
 mean=statistics.fmean(rows);se=statistics.stdev(rows)/math.sqrt(len(rows))
 return {"orders":len(rows),"positive_orders":sum(x>0 for x in rows),"A_mean":mean,"A_ci95":[mean-1.96*se,mean+1.96*se],"passed":mean-1.96*se>0}
def run():
 Path("artifacts").mkdir(exist_ok=True);out={n:family(n,t) for n,t in FAMILIES.items()};out["replicated_across_families"]=all(x["passed"] for x in out.values())
 Path("artifacts/A_replication_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
