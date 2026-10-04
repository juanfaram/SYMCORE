#!/usr/bin/env python3
"""Out-of-distribution generalization arena: unseen combinations are held out until evaluation."""
import json,random
from pathlib import Path
from interaction_learner import InteractionLearner
def run(seed=303):
 rng=random.Random(seed);actions=["fast","balanced","deep","creative"];a=InteractionLearner(actions,.08,seed,"artifacts/ood.jsonl")
 domains=["code","analysis","writing","forecast"];diffs=["simple","normal","hard"]
 held={("writing","hard"),("forecast","simple"),("code","normal")}
 def target(d,q):
  if d=="writing":return "creative"
  if q=="hard" or d=="code":return "deep"
  return "fast" if q=="simple" else "balanced"
 train=[(d,q) for d in domains for q in diffs if (d,q) not in held]
 for i in range(12000):
  d,q=rng.choice(train);c={"task":"solve","domain":d,"difficulty":q};x=a.choose(c);a.feedback(x,c,1 if x==target(d,q) else -.3,{"split":"train"})
 seen=unseen=0;sn=un=0
 for _ in range(3000):
  d,q=rng.choice([(d,q) for d in domains for q in diffs]);c={"task":"solve","domain":d,"difficulty":q};ok=a.choose(c)==target(d,q)
  if (d,q) in held:unseen+=ok;un+=1
  else:seen+=ok;sn+=1
 out={"seen_accuracy":round(seen/sn,4),"unseen_combo_accuracy":round(unseen/un,4),"held_out":[list(x) for x in sorted(held)]}
 Path("artifacts/ood_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
