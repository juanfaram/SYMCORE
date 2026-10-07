#!/usr/bin/env python3
"""Validity audit for Host3 only. Does not modify Pi, environment, seeds, or endpoints."""
import json,random,statistics
from pathlib import Path
from host3_bandit import Detector,env_probs
OPS=("fast","slow","reset")
def detector_accuracy(seed,T=720):
 d=Detector();rng=random.Random(seed);correct=total=0;prev_best=None
 for t in range(T):
  ps=env_probs(t,seed);best=max(range(3),key=ps.__getitem__);truth="drift" if prev_best is not None and best!=prev_best else "stable"
  pred=d.gap();correct+=pred==truth;total+=1
  arm=rng.randrange(3);reward=int(rng.random()<ps[arm]);d.observe(abs(reward-.5));prev_best=best
 return correct/total
def operator_oracle(seed,op,T=720):
 rng=random.Random(seed);q=[.5]*3;n=[0]*3;reg=0.
 for t in range(T):
  ps=env_probs(t,seed);eps={"fast":.28,"slow":.08,"reset":.16}[op]
  arm=rng.randrange(3) if rng.random()<eps else max(range(3),key=q.__getitem__);reward=int(rng.random()<ps[arm]);reg+=max(ps)-ps[arm]
  lr={"fast":.22,"slow":.06,"reset":.12}[op];q[arm]=(1-lr)*q[arm]+lr*reward
 return reg
def run():
 seeds=range(200);acc=[detector_accuracy(s) for s in seeds];means={op:statistics.fmean(operator_oracle(s,op) for s in seeds) for op in OPS}
 out={"detector_accuracy_mean":statistics.fmean(acc),"detector_valid_ge_70pct":statistics.fmean(acc)>=.70,
      "operator_regret_means":means,"operator_structure_nontrivial":max(means.values())-min(means.values())>1.0,
      "note":"accuracy against one-step regime-transition label; operator audit checks non-equivalence, not per-regime specialization"}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/host3_validity.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
