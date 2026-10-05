#!/usr/bin/env python3
"""Adaptive-prior Π: can accumulated experience accelerate online learning without rigidity?"""
import json,random,statistics
from pathlib import Path
from operator_policy import OperatorPolicy
BASE=[
 ("drift","memory",.78,"learning",.62),("composition","structure",.76,"representation",.61),
 ("sparse","features",.75,"representation",.58),("novel","representation",.74,"features",.59),
 ("overfit","parameters",.73,"learning",.57),("objective_mismatch","objectives",.78,"learning",.55)]
# After half the exam utilities drift: the old best remains useful but the former runner-up becomes best.
def prob(gap,op,phase):
 row=next(x for x in BASE if x[0]==gap);a,pa,b,pb=row[1:]
 if phase==0:return pa if op==a else (pb if op==b else .10)
 return .54 if op==a else (.78 if op==b else .10)
HISTORY=[(g,a,1) for g,a,pa,b,pb in BASE]+[(g,b,1) for g,a,pa,b,pb in BASE]+[(g,"memory",0) for g,a,pa,b,pb in BASE if "memory" not in (a,b)]
def compare(x,y):
 d=[b-a for a,b in zip(x,y)];m=statistics.fmean(d);se=statistics.stdev(d)/(len(d)**.5)
 return {"delta":m,"ci95":[m-1.96*se,m+1.96*se]}
def arm(seed,mode,strength=0.,draws=300):
 rng=random.Random(seed);p=OperatorPolicy(history_strength=strength)
 if mode=="adaptive":
  for gap,op,s in HISTORY:p.update(gap,op,s,weight=strength)
 score=0.
 for i in range(draws):
  gap=BASE[i%len(BASE)][0];phase=int(i>=draws//2);op=p.choose(gap,rng);ok=rng.random()<prob(gap,op,phase);score+=ok
  if mode in ("online","adaptive"):p.update(gap,op,ok)
 return score/draws
def run(seeds=range(100)):
 p05=[arm(s,"online") for s in seeds]
 strengths=(.10,.25,.50,1.0,2.0,4.0);curve=[]
 for h in strengths:
  xs=[arm(s,"adaptive",h) for s in seeds];cmp=compare(p05,xs)
  curve.append({"history_strength":h,"mean":statistics.fmean(xs),"vs_Pi05":cmp})
 positive=[x for x in curve if x["vs_Pi05"]["ci95"][0]>0]
 # Robustness, not best-point optimization: require at least half the preregistered strengths to win.
 out={"schema":"symcore.pi_adaptive.v1","seeds":len(p05),"Pi05_online_mean":statistics.fmean(p05),
      "history_strength_curve":curve,"positive_strengths":len(positive),"required_positive_strengths":len(strengths)//2,
      "utility_drift_at_half":True,"same_online_budget":True,
      "passed":len(positive)>=len(strengths)//2}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/pi_adaptive_report.json").write_text(json.dumps(out,indent=2))
 print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
