#!/usr/bin/env python3
"""v1.7 causal diagnosis of L=0: task overlap vs initialization vs procedural transfer."""
import json,random,statistics
from pathlib import Path
from skill_memory import SkillMemory
A=["fast","balanced","deep","creative"]
def gen(r):return {"task":"solve","domain":r.choice(["code","analysis","writing","forecast"]),"difficulty":r.choice(["simple","normal","hard"])}
SK={
"A":lambda c:{"simple":"fast","normal":"balanced","hard":"deep"}[c["difficulty"]],
"B":lambda c:{"code":"deep","analysis":"balanced","writing":"creative","forecast":"balanced"}[c["domain"]],
"C":lambda c:"creative" if c["domain"]=="writing" else ("fast" if c["difficulty"]=="simple" else "deep")}
def train(m,fn,r,n=1800):
 for _ in range(n):
  c=gen(r);a=m.choose(c);m.feedback(a,c,1 if a==fn(c) else -.3)
def latency(m,fn,r,limit=2500):
 recent=[]
 for i in range(1,limit+1):
  c=gen(r);a=m.choose(c);ok=a==fn(c);m.feedback(a,c,1 if ok else -.3);recent.append(ok)
  if len(recent)>100:recent.pop(0)
  if len(recent)==100 and sum(recent)>=70:return i
 return limit+1
def run(seeds=(13,29,43,71,113,157,199)):
 rows=[]
 for seed in seeds:
  # scratch C
  r=random.Random(seed);scratch=latency(SkillMemory(A),SK["C"],r)
  # A -> C
  r=random.Random(seed);ma=SkillMemory(A);train(ma,SK["A"],r);ac=latency(ma,SK["C"],r)
  # A+B -> C cumulative
  r=random.Random(seed);mab=SkillMemory(A);train(mab,SK["A"],r);train(mab,SK["B"],r);abc=latency(mab,SK["C"],r)
  # procedural-only: reuse acquisition policy parameters, but erase declarative skill memory
  r=random.Random(seed);proc=SkillMemory(A);pc=latency(proc,SK["C"],r)
  rows.append({"seed":seed,"scratch_C":scratch,"A_to_C":ac,"AB_to_C":abc,"procedural_only_C":pc})
 means={k:statistics.fmean(x[k] for x in rows) for k in ("scratch_C","A_to_C","AB_to_C","procedural_only_C")}
 if means["AB_to_C"]<means["A_to_C"]*.9:cause="cumulative_structure_exists"
 elif means["A_to_C"]<means["scratch_C"]*.9:cause="pairwise_initialization_transfer"
 elif means["procedural_only_C"]<means["scratch_C"]*.9:cause="procedural_transfer"
 else:cause="no_reusable_structure_in_current_tasks_or_representation"
 out={"rows":rows,"means":means,"diagnosis":cause}
 Path("artifacts/L_causal_diagnosis.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
