#!/usr/bin/env python3
"""Directional transfer matrix: A->B is measured separately from B->A."""
import json,random
from pathlib import Path
from representation import StructuralRepresentation
from representation_learner import RepresentationLearner
A=["fast","balanced","deep","creative"]
SK={
"difficulty":lambda c:{"simple":"fast","normal":"balanced","hard":"deep"}[c["difficulty"]],
"domain":lambda c:{"code":"deep","analysis":"balanced","writing":"creative","forecast":"balanced"}[c["domain"]],
"hybrid":lambda c:"creative" if c["domain"]=="writing" else ("fast" if c["difficulty"]=="simple" else "deep")}
def gen(r):return {"task":"solve","domain":r.choice(["code","analysis","writing","forecast"]),"difficulty":r.choice(["simple","normal","hard"])}
def train(agent,fn,r,n):
 for _ in range(n):
  c=gen(r);a=agent.choose(c);agent.feedback(a,c,1 if a==fn(c) else -.3)
def latency(agent,fn,r,limit=3000):
 recent=[]
 for i in range(1,limit+1):
  c=gen(r);a=agent.choose(c);ok=a==fn(c);agent.feedback(a,c,1 if ok else -.3);recent.append(ok)
  if len(recent)>100:recent.pop(0)
  if len(recent)==100 and sum(recent)>=70:return i
 return limit+1
def run(seeds=(5,17,31,59,97)):
 out={}
 for src in SK:
  for dst in SK:
   if src==dst:continue
   vals=[]
   for seed in seeds:
    r=random.Random(seed);rep=StructuralRepresentation();ag=RepresentationLearner(A,rep);train(ag,SK[src],r,2500);vals.append(latency(ag,SK[dst],r))
   out[f"{src}->{dst}"]=vals
 Path("artifacts/transfer_matrix.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
