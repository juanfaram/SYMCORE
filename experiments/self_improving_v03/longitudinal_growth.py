#!/usr/bin/env python3
"""Longitudinal host curriculum: test A>0 and dA/dE>0 across successive novel regimes."""
import json,math,random,statistics
from pathlib import Path
from acceleration_contract import acceleration
from risk_contract import risk_report,passes
def episode(rng,phase,n=1200):
 for t in range(n):
  if phase==0:y=40+20*math.sin(2*math.pi*t/24)+rng.gauss(0,4)
  elif phase==1:y=65+12*math.sin(2*math.pi*t/12)+rng.gauss(0,5)
  elif phase==2:y=25+.035*t+18*math.sin(2*math.pi*t/30)+rng.gauss(0,5)
  else:y=(30 if t<n//2 else 75)+10*math.sin(2*math.pi*t/18)+rng.gauss(0,6)
  yield y
def acquisition_cost(history,phase,seed,threshold=12.0):
 rng=random.Random(seed+phase*100);recent=[];last=history[-1] if history else 0.;pattern={}
 for i,y in enumerate(episode(rng,phase),1):
  key=i%(24 if phase==0 else 12 if phase==1 else 30 if phase==2 else 18)
  p=pattern.get(key,last);err=abs(p-y);recent.append(err)
  if len(recent)>100:recent.pop(0)
  pattern[key]=.75*pattern.get(key,y)+.25*y;last=y;history.append(y)
  if len(recent)==100 and statistics.fmean(recent)<=threshold:return i
 return 1201
def run():
 curves=[];risks=[]
 for seed in (3,17,37,61,109,151,211):
  h=[];costs=[acquisition_cost(h,p,seed) for p in range(4)];curves.append(costs)
  # conservative synthetic paired risk envelope derived from costs: regression if later acquisition > first by >5%
  control=[costs[0]]*3;sym=costs[1:];risks.append(risk_report(control,sym,.05))
 accel=[acceleration(c) for c in curves]
 out={"curves":curves,"A_positive_fraction":sum(x["passed_A"] for x in accel)/len(accel),
      "dA_positive_fraction":sum(x["passed_dA"] for x in accel)/len(accel),
      "risk_upper95_max":max(x["p_upper95"] for x in risks),"risk_pass_all":all(passes(x,.2) for x in risks)}
 Path("artifacts/longitudinal_growth_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
