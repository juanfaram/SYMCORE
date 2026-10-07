#!/usr/bin/env python3
"""Generic causal V_learn(E,h) surface runner."""
from __future__ import annotations
import copy,statistics,math
from VLEARN_PREREGISTRATION import *
def evaluate_surface(rows,make_learner,normalize,target_fn):
 # rows are chronological external observations; no shuffling.
 learner=make_learner();surfaces={};pending={}
 max_needed=max(FREEZE_POINTS)+max(HORIZONS)
 for t,raw in enumerate(rows,1):
  y=target_fn(raw)
  if y is None or (isinstance(y,float) and math.isnan(y)):continue
  r=normalize(raw)
  # Before update, create counterfactual pairs at exact valid-experience index.
  e=getattr(learner,"_vlearn_e",0)+1;learner._vlearn_e=e
  if e in FREEZE_POINTS:
   pending[e]={"live":copy.deepcopy(learner),"frozen":copy.deepcopy(learner),"errs":{h:[[],[]] for h in HORIZONS}}
  # Main learner keeps learning to seed later freeze points.
  learner.step(r,float(y),True)
  for E,p in list(pending.items()):
   age=e-E
   if age<=0 or age>max(HORIZONS):continue
   le,_=p["live"].step(r,float(y),True);fe,_=p["frozen"].step(r,float(y),False)
   for h in HORIZONS:
    if age<=h:p["errs"][h][0].append(le);p["errs"][h][1].append(fe)
   if age==max(HORIZONS):
    surfaces[E]={h:statistics.fmean(p["errs"][h][1])-statistics.fmean(p["errs"][h][0]) for h in HORIZONS}
 return surfaces
def summarize(surface):
 positive={E:{h:(v>0) for h,v in hs.items()} for E,hs in surface.items()}
 passing_E=[E for E,hs in positive.items() if sum(hs.values())>=MIN_POSITIVE_HORIZONS]
 # finite differences describe shape; no claim of derivative sign is preregistered as CL1/CL2.
 dE=[]
 for h in HORIZONS:
  pts=[(E,surface[E][h]) for E in sorted(surface)]
  dE.extend((pts[i+1][1]-pts[i][1])/(pts[i+1][0]-pts[i][0]) for i in range(len(pts)-1))
 dh=[]
 for E in sorted(surface):
  pts=[(h,surface[E][h]) for h in HORIZONS]
  dh.extend((pts[i+1][1]-pts[i][1])/(pts[i+1][0]-pts[i][0]) for i in range(len(pts)-1))
 return {"passing_freeze_points":len(passing_E),"passed":len(passing_E)>=MIN_POSITIVE_FREEZE_POINTS,
         "dV_dE_mean":statistics.fmean(dE) if dE else None,"dV_dh_mean":statistics.fmean(dh) if dh else None}
