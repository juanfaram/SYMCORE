#!/usr/bin/env python3
"""Exhaustive factorial audit for z9/z10/z14 with paired LOHO value and correlation."""
from __future__ import annotations
import json,statistics,math
from itertools import combinations
from SURVIVOR_FACTORIAL_CONTRACT import *
from pi_learn import PiLearn,features
from pi_learn_value import score
def pearson(xs,ys):
 if len(xs)<2:return 0.
 xm=statistics.fmean(xs);ym=statistics.fmean(ys);den=math.sqrt(sum((x-xm)**2 for x in xs)*sum((y-ym)**2 for y in ys))
 return sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/den if den else 0.
def eval_subset(data,keep):
 out={}
 for held in data:
  train=[r for h,rows in data.items() if h!=held for r in rows];test=data[held];p=PiLearn(k=min(12,max(3,len(train))),margin=.5)
  for r in train:p.observe(features(r["E"],r["h"],tuple(r["z"][i] for i in keep)),r["V"])
  dec=[]
  for r in test:
   a,_=p.decide(features(r["E"],r["h"],tuple(r["z"][i] for i in keep)));dec.append((a,r["V"]))
  out[held]=score(dec)
 return out
def audit(data):
 base=eval_subset(data,())
 subsets={};best=None
 for keep in SUBSETS:
  sc=eval_subset(data,keep);wins=[]
  for h in data:
   wins.append(sc[h]["value_capture"]>base[h]["value_capture"] and sc[h]["mean_regret"]<=base[h]["mean_regret"])
  agg=statistics.fmean(sc[h]["value_capture"] for h in data)
  subsets[str(keep)]={"scores":sc,"transfer_wins":sum(wins),"mean_value_capture":agg}
  candidate=(sum(wins),agg,keep)
  if best is None or candidate>best:best=candidate
 corr={}
 for h,rows in data.items():
  corr[h]={}
  for a,b in combinations(SURVIVORS,2):corr[h][f"{a}:{b}"]=pearson([r["z"][a] for r in rows],[r["z"][b] for r in rows])
 return {"baseline":base,"subsets":subsets,"correlations":corr,
         "best":{"keep":best[2],"transfer_wins":best[0],"mean_value_capture":best[1]},
         "passes_transfer_gate":best[0]>=TRANSFER_HOSTS_GTE}
