#!/usr/bin/env python3
"""Longitudinal meta-learning test: does Π improve after observing its own prior generation?"""
import json,random,statistics
from pathlib import Path
from operator_policy import OperatorPolicy
OPS=("parameters","features","memory","representation","structure","objectives","learning")
GAPS=("drift","composition","sparse","novel","overfit","objective_mismatch")
# Fixed latent utility landscape; policy never sees these probabilities.
P={
"drift":{"memory":.72,"learning":.62},"composition":{"structure":.70,"representation":.64},
"sparse":{"features":.71,"representation":.58},"novel":{"representation":.68,"features":.61},
"overfit":{"parameters":.69,"learning":.57},"objective_mismatch":{"objectives":.73,"learning":.55}}
def prob(g,o):return P[g].get(o,.10)
def generation(policy,rng,n=240,learn=True):
 score=0
 for i in range(n):
  g=GAPS[i%len(GAPS)];op=policy.choose(g,rng);ok=rng.random()<prob(g,op);score+=ok
  if learn:policy.update(g,op,ok)
 return score/n
def snapshot(policy):
 q=OperatorPolicy()
 for g in GAPS:
  for op in OPS:
   q.success[g][op]=policy.success[g][op];q.trials[g][op]=policy.trials[g][op]
 return q
def naive_generation(counts,rng,n=240,learn=True):
 score=0
 for i in range(n):
  g=GAPS[i%len(GAPS)];best=max(OPS,key=lambda o:counts[g][o][0]/max(1,counts[g][o][1]));op=best if counts[g][best][1] else rng.choice(OPS);ok=rng.random()<prob(g,op);score+=ok
  if learn:counts[g][op][0]+=ok;counts[g][op][1]+=1
 return score/n

def run(seeds=range(100),generations=6):
 trajectories=[];naive=[]
 for seed in seeds:
  rng=random.Random(seed);p=OperatorPolicy();row=[]
  for gen in range(generations):
   # Freeze Π_t for evaluation, then let the live policy learn only from generation t.
   frozen=snapshot(p);score=generation(frozen,random.Random(seed*1000+gen),learn=False);row.append(score)
   generation(p,rng,learn=True)
  trajectories.append(row)
  counts={g:{o:[0,0] for o in OPS} for g in GAPS};nr=[]
  for gen in range(generations):
   frozen={g:{o:list(v) for o,v in d.items()} for g,d in counts.items()};nr.append(naive_generation(frozen,random.Random(seed*2000+gen),learn=False));naive_generation(counts,random.Random(seed+gen),learn=True)
  naive.append(nr)
 means=[statistics.fmean(r[g] for r in trajectories) for g in range(generations)];naive_means=[statistics.fmean(r[g] for r in naive) for g in range(generations)];increments=[means[i]-means[i-1] for i in range(1,generations)]
 slopes=[]
 for r in trajectories:
  xbar=(generations-1)/2;ybar=statistics.fmean(r)
  slopes.append(sum((i-xbar)*(y-ybar) for i,y in enumerate(r))/sum((i-xbar)**2 for i in range(generations)))
 m=statistics.fmean(slopes);se=statistics.stdev(slopes)/(len(slopes)**.5)
 first_last=[r[-1]-r[0] for r in trajectories];fl=statistics.fmean(first_last);flse=statistics.stdev(first_last)/(len(first_last)**.5)
 out={"schema":"symcore.pi_longitudinal.v1","seeds":len(trajectories),"generations":generations,
      "generation_means":means,"generation_increments":increments,"naive_generation_means":naive_means,"constant_training_budget_per_generation":240,"slope_mean":m,"slope_ci95":[m-1.96*se,m+1.96*se],
      "Pi_last_minus_Pi_first":fl,"first_last_ci95":[fl-1.96*flse,fl+1.96*flse],
      "passed":m>0 and m-1.96*se>0 and fl-1.96*flse>0}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/pi_longitudinal_report.json").write_text(json.dumps(out,indent=2))
 print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
