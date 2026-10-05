#!/usr/bin/env python3
"""Permuted longitudinal curriculum: order-robust evidence for A and dA/dE."""
import itertools,json,random,statistics,math
from pathlib import Path
from acceleration_contract import acceleration
def episode(rng,kind,n=700):
 for t in range(n):
  if kind=="periodic":y=40+20*math.sin(2*math.pi*t/24)+rng.gauss(0,4)
  elif kind=="fast":y=55+14*math.sin(2*math.pi*t/8)+rng.gauss(0,5)
  elif kind=="trend":y=20+.04*t+12*math.sin(2*math.pi*t/30)+rng.gauss(0,5)
  else:y=(25 if t<n//2 else 70)+rng.gauss(0,6)
  yield y
def acquire(kind,seed,threshold=13):
 rng=random.Random(seed);mem={};recent=[]
 period={"periodic":24,"fast":8,"trend":30,"regime":18}[kind]
 for i,y in enumerate(episode(rng,kind),1):
  k=i%period;p=mem.get(k,y);recent.append(abs(p-y))
  if len(recent)>80:recent.pop(0)
  mem[k]=.75*mem.get(k,y)+.25*y
  if len(recent)==80 and statistics.fmean(recent)<=threshold:return i
 return 701
def run():
 kinds=("periodic","fast","trend","regime");seeds=(5,17,41,73,109,163,223);rows=[]
 for oi,order in enumerate(itertools.permutations(kinds)):
  curves=[]
  for s in seeds:curves.append([acquire(k,s+oi*1000+j*100) for j,k in enumerate(order)])
  ev=[acceleration(c) for c in curves]
  rows.append({"order":order,"curves":curves,"A_fraction":sum(x["passed_A"] for x in ev)/len(ev),"dA_fraction":sum(x["passed_dA"] for x in ev)/len(ev)})
 out={"orders":rows,"orders_with_A":sum(x["A_fraction"]>.5 for x in rows),"orders_with_dA":sum(x["dA_fraction"]>.5 for x in rows),"total_orders":len(rows)}
 Path("artifacts/longitudinal_permutations.json").write_text(json.dumps(out,indent=2));print(json.dumps({"orders_with_A":out["orders_with_A"],"orders_with_dA":out["orders_with_dA"],"total_orders":len(rows)},indent=2));return out
if __name__=="__main__":run()
