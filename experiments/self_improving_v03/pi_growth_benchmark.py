#!/usr/bin/env python3
"""Three-arm test: prior experience vs online learning vs uniform generation."""
import json,random,statistics
from pathlib import Path
from operator_policy import OperatorPolicy
# Multiple opportunity profiles per gap; rewards overlap so no operator is a trivial label lookup.
TASKS=[
 ("drift-fast","drift",{"memory":.85,"learning":.70,"parameters":.35}),
 ("drift-slow","drift",{"learning":.80,"memory":.55,"representation":.30}),
 ("compose-route","composition",{"structure":.80,"representation":.65,"memory":.30}),
 ("compose-latent","composition",{"representation":.85,"structure":.55,"features":.30}),
 ("sparse-new","sparse",{"features":.80,"representation":.55,"parameters":.25}),
 ("novel-domain","novel",{"representation":.75,"features":.60,"learning":.35}),
 ("overfit","overfit",{"parameters":.75,"learning":.55,"memory":.25}),
 ("objective-shift","objective_mismatch",{"objectives":.85,"learning":.50,"structure":.25})]
HISTORY=[("drift","memory",1),("drift","learning",1),("drift","parameters",0),
 ("composition","structure",1),("composition","representation",1),("composition","memory",0),
 ("sparse","features",1),("novel","representation",1),("overfit","parameters",1),
 ("objective_mismatch","objectives",1)]
def outcome(task,op,rng):
 p=task[2].get(op,.08);return rng.random()<p
def arm(seed,mode,draws=240):
 rng=random.Random(seed);p=OperatorPolicy()
 if mode=="prior":
  for gap,op,s in HISTORY:p.update(gap,op,s)
 score=0.
 for i in range(draws):
  task=TASKS[i%len(TASKS)];gap=task[1];op=p.choose(gap,rng);ok=outcome(task,op,rng);score+=ok
  if mode=="online":p.update(gap,op,ok)
 return score/draws
def compare(x,y):
 d=[b-a for a,b in zip(x,y)];m=statistics.fmean(d);se=statistics.stdev(d)/(len(d)**.5)
 return {"delta":m,"ci95":[m-1.96*se,m+1.96*se]}
def run(seeds=range(80)):
 p0=[arm(s,"uniform") for s in seeds]
 p05=[arm(s,"online") for s in seeds]
 p1=[arm(s,"prior") for s in seeds]
 prior_vs_uniform=compare(p0,p1);prior_vs_online=compare(p05,p1)
 out={"schema":"symcore.pi_growth.v2","seeds":len(p0),"gap_types":len(set(t[1] for t in TASKS)),
      "Pi0_uniform_mean":statistics.fmean(p0),"Pi05_online_mean":statistics.fmean(p05),"Pi1_prior_mean":statistics.fmean(p1),
      "Pi1_vs_Pi0":prior_vs_uniform,"Pi1_vs_Pi05":prior_vs_online,
      "passed":prior_vs_uniform["ci95"][0]>0 and prior_vs_online["ci95"][0]>0}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/pi_growth_report.json").write_text(json.dumps(out,indent=2))
 print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
