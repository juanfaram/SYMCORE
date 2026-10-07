#!/usr/bin/env python3
"""Leakage-free full FunctionalGeometry snapshot collector for V_learn decisions."""
from __future__ import annotations
import copy,math,statistics
from functional_geometry import FunctionalGeometry
from VLEARN_PREREGISTRATION import FREEZE_POINTS,HORIZONS
def collect_full_geometry(rows,make_learner,normalize,target_fn,signal_fn):
 learner=make_learner();geom=FunctionalGeometry(window=64);pending={};out=[];E=0;maxh=max(HORIZONS)
 for raw in rows:
  y=target_fn(raw)
  if y is None or (isinstance(y,float) and math.isnan(y)):continue
  E+=1;r=normalize(raw);y=float(y)
  # Snapshot BEFORE current target/outcome. signal_fn may use only contemporaneously observable covariates.
  if E in FREEZE_POINTS:
   pending[E]={"z":geom.vector(),"live":copy.deepcopy(learner),"frozen":copy.deepcopy(learner),
               "errs":{h:[[],[]] for h in HORIZONS}}
  ll,_=learner.step(r,y,True)
  geom.observe(signal_fn(r,ll))
  for e,p in list(pending.items()):
   age=E-e
   if age<=0 or age>maxh:continue
   le,_=p["live"].step(r,y,True);fe,_=p["frozen"].step(r,y,False)
   for h in HORIZONS:
    if age<=h:p["errs"][h][0].append(le);p["errs"][h][1].append(fe)
   if age==maxh:
    for h in HORIZONS:
     live,frozen=p["errs"][h];out.append({"E":e,"h":h,"z":p["z"],"V":statistics.fmean(frozen)-statistics.fmean(live)})
    del pending[e]
 return out
