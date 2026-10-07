#!/usr/bin/env python3
"""v1.5: permutation test for learning acceleration L(t)."""
import itertools,json,random,statistics
from pathlib import Path
from skill_memory import SkillMemory
from transfer_prior import TransferPrior
ACTIONS=["fast","balanced","deep","creative"]
SKILLS={
"difficulty":lambda c:{"simple":"fast","normal":"balanced","hard":"deep"}[c["difficulty"]],
"domain":lambda c:{"code":"deep","analysis":"balanced","writing":"creative","forecast":"balanced"}[c["domain"]],
"hybrid":lambda c:"creative" if c["domain"]=="writing" else ("fast" if c["difficulty"]=="simple" else "deep")}
def ctx(r):return {"task":"solve","domain":r.choice(["code","analysis","writing","forecast"]),"difficulty":r.choice(["simple","normal","hard"])}
def acquire(m,p,fn,r,max_steps=3500):
 recent=[]
 for i in range(1,max_steps+1):
  c=ctx(r);p.seed(m,c,1);a=m.choose(c);ok=a==fn(c);rew=1 if ok else -.3;m.feedback(a,c,rew);p.observe(c,a,rew);recent.append(ok)
  if len(recent)>100:recent.pop(0)
  if len(recent)==100 and sum(recent)>=75:return i
 return max_steps+1
def run(seeds=(7,19,41,67,101)):
 rows=[]
 for order in itertools.permutations(SKILLS):
  costs=[]
  for seed in seeds:
   r=random.Random(seed);m=SkillMemory(ACTIONS);p=TransferPrior(ACTIONS);seq=[]
   for name in order:seq.append(acquire(m,p,SKILLS[name],r))
   costs.append(seq)
  means=[statistics.fmean(x[i] for x in costs) for i in range(3)]
  L=(means[0]-means[-1])/max(1,means[0])
  rows.append({"order":order,"mean_costs":means,"L_relative":round(L,4),"positive":L>0})
 vals=[x["L_relative"] for x in rows]
 out={"orders":rows,"positive_fraction":sum(x["positive"] for x in rows)/len(rows),"median_L":statistics.median(vals),"min_L":min(vals),"max_L":max(vals),"seeds":list(seeds)}
 Path("artifacts/L_permutation_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
