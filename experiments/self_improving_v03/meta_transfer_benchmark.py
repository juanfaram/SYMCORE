#!/usr/bin/env python3
"""Multi-seed forward-transfer benchmark with reachable target."""
import json,random,statistics
from pathlib import Path
from skill_memory import SkillMemory
from transfer_prior import TransferPrior
def trial(seed,experienced):
 rng=random.Random(seed);actions=["fast","balanced","deep","creative"];m=SkillMemory(actions);p=TransferPrior(actions)
 def gen():return {"task":"solve","domain":rng.choice(["code","writing","analysis"]),"difficulty":rng.choice(["simple","normal","hard"])}
 def pre(c):return {"simple":"fast","normal":"balanced","hard":"deep"}[c["difficulty"]]
 def target(c):return "creative" if c["domain"]=="writing" else ("deep" if c["difficulty"]=="hard" else "balanced")
 if experienced:
  for _ in range(2500):
   c=gen();a=pre(c);p.observe(c,a,1)
 # target uses factor priors only as initialization; subsequent evidence can override
 recent=[]
 for i in range(1,4001):
  c=gen()
  if experienced and i<=300:p.seed(m,c,2)
  a=m.choose(c);ok=a==target(c);m.feedback(a,c,1 if ok else -.3);recent.append(ok)
  if len(recent)>100:recent.pop(0)
  if len(recent)==100 and sum(recent)>=75:return i
 return 4001
def run():
 seeds=[3,11,29,47,83,101,131];scratch=[trial(s,False) for s in seeds];exp=[trial(s,True) for s in seeds]
 sm=statistics.fmean(scratch);em=statistics.fmean(exp);ft=(sm-em)/sm
 out={"seeds":seeds,"scratch":scratch,"experienced":exp,"scratch_mean":sm,"experienced_mean":em,"forward_transfer":round(ft,4)}
 Path("artifacts/meta_transfer_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
