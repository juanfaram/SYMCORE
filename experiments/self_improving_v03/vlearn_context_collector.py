#!/usr/bin/env python3
"""Prospective-in-time collector for (E,z_H^learn,h,V_learn) on opened development hosts."""
from __future__ import annotations
import copy,math,statistics
from learning_state import LearningState
from VLEARN_PREREGISTRATION import FREEZE_POINTS,HORIZONS
def collect(rows,make_learner,normalize,target_fn):
 learner=make_learner();state=LearningState();pending={};out=[];E=0;maxh=max(HORIZONS)
 for raw in rows:
  y=target_fn(raw)
  if y is None or (isinstance(y,float) and math.isnan(y)):continue
  E+=1;r=normalize(raw);y=float(y)
  # z is captured BEFORE observing target/loss at E: no leakage from current or future outcome.
  if E in FREEZE_POINTS:
   pending[E]={"z":state.vector(),"live":copy.deepcopy(learner),"frozen":copy.deepcopy(learner),
               "errs":{h:[[],[]] for h in HORIZONS}}
  main_loss,_=learner.step(r,y,True);state.observe(main_loss)
  for e,p in list(pending.items()):
   age=E-e
   if age<=0 or age>maxh:continue
   ll,_=p["live"].step(r,y,True);fl,_=p["frozen"].step(r,y,False)
   for h in HORIZONS:
    if age<=h:p["errs"][h][0].append(ll);p["errs"][h][1].append(fl)
   if age==maxh:
    for h in HORIZONS:
     live,frozen=p["errs"][h]
     out.append({"E":e,"h":h,"z":p["z"],"V":statistics.fmean(frozen)-statistics.fmean(live)})
    del pending[e]
 return out
