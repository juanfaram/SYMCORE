#!/usr/bin/env python3
"""Adversarial interaction arena: noise, delayed feedback, regime switches, rare contexts."""
import json,random
from pathlib import Path
from interaction_learner import InteractionLearner
def run(seed=117,n=20000):
 rng=random.Random(seed);a=InteractionLearner(["fast","balanced","deep","creative"],.12,seed,"artifacts/adversarial.jsonl")
 domains=["code"]*4+["analysis"]*3+["writing"]*2+["rare"];diffs=["simple","normal","hard"];correct=0;recent=[]
 for i in range(n):
  d=rng.choice(domains);q=rng.choice(diffs);ctx={"task":"solve","domain":d,"difficulty":q}
  phase=(i//4000)%4
  maps=[
   lambda:{"simple":"fast","normal":"balanced","hard":"deep"}[q],
   lambda:{"code":"deep","analysis":"balanced","writing":"creative","rare":"fast"}[d],
   lambda:"creative" if d=="writing" else ("deep" if q=="hard" else "balanced"),
   lambda:"fast" if d=="rare" else ("deep" if d in ("code","analysis") else "creative")
  ];target=maps[phase]()
  action=a.choose(ctx);ok=action==target;reward=(1 if ok else -.35)+rng.gauss(0,.12)
  if rng.random()<.08:reward*=-1
  a.feedback(action,ctx,reward,{"phase":phase});correct+=ok;recent.append(ok)
  if len(recent)>1000:recent.pop(0)
 out={"interactions":n,"overall_accuracy":round(correct/n,4),"final_1000_accuracy":round(sum(recent)/len(recent),4)}
 Path("artifacts/adversarial_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
