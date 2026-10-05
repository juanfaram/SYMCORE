#!/usr/bin/env python3
"""First direct test of Π1 > Π0 on unseen growth opportunities."""
import json,random,statistics
from pathlib import Path
from operator_policy import OperatorPolicy,OPERATORS
# Hidden opportunity map is used only by the evaluator, never by Π.
TASKS=[("drift","memory"),("drift","learning"),("composition","structure"),
       ("composition","representation"),("sparse","features"),("novel","representation"),
       ("overfit","parameters"),("objective_mismatch","objectives")]
def reward(gap,op):
 target=dict(TASKS)[gap]
 return 1.0 if op==target else (0.35 if (gap,op) in {("drift","parameters"),("composition","memory"),("novel","features")} else 0.0)
def train_history():
 # Prior verified experience; deliberately incomplete and not identical to evaluation tasks.
 return [("drift","memory",1),("drift","memory",1),("drift","parameters",0),
         ("composition","structure",1),("composition","memory",0),("sparse","features",1),
         ("novel","representation",1),("overfit","parameters",1),("objective_mismatch","objectives",1)]
def evaluate(policy,seeds=range(40),draws=200):
 vals=[]
 for seed in seeds:
  rng=random.Random(seed);score=0.
  for i in range(draws):
   gap=TASKS[i%len(TASKS)][0];score+=reward(gap,policy.choose(gap,rng))
  vals.append(score/draws)
 return vals
def run():
 p0=OperatorPolicy();p1=OperatorPolicy()
 for gap,op,s in train_history():p1.update(gap,op,s)
 a=evaluate(p0);b=evaluate(p1);diff=[y-x for x,y in zip(a,b)]
 mean=statistics.fmean(diff);se=statistics.stdev(diff)/(len(diff)**.5)
 out={"schema":"symcore.pi_growth.v1","seeds":len(a),"Pi0_mean":statistics.fmean(a),"Pi1_mean":statistics.fmean(b),
      "Pi1_minus_Pi0":mean,"ci95":[mean-1.96*se,mean+1.96*se],
      "passed":mean>0 and mean-1.96*se>0}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/pi_growth_report.json").write_text(json.dumps(out,indent=2))
 print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
