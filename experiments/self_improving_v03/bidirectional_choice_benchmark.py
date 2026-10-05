#!/usr/bin/env python3
"""Preregistered 460 benchmark: can one frozen Pi choose evolutionary operations from functional state?"""
from __future__ import annotations
import json,math,random,statistics
from pathlib import Path
OPS=("create","modify","combine","freeze","prune","restore","noop")
# Functional state only. No scenario name or optimal-op label is exposed to Pi.
FEATURES=("capacity_gap","plasticity","redundancy","load","archive_relevance","frontier_pressure","interference")
SCENARIOS={
 "redundant_loaded":(.20,.45,.82,.88,.10,.25,.40),
 "capacity_plastic":(.84,.86,.18,.35,.05,.72,.28),
 "archive_relevant":(.52,.55,.30,.48,.91,.61,.32),
 "stable_unpressured":(.16,.32,.22,.18,.08,.10,.18)}
# Operation effect vectors in the same functional coordinates; these define mechanisms, not answers.
EFFECTS={
 "create":(-.70,+.20,+.18,+.42,0.,-.45,+.12),
 "modify":(-.25,+.12,-.05,+.05,0.,-.18,-.18),
 "combine":(-.38,+.08,+.30,+.22,0.,-.28,+.22),
 "freeze":(+.08,-.18,-.12,-.20,0.,+.02,-.12),
 "prune":(+.12,-.08,-.62,-.58,0.,+.05,-.30),
 "restore":(-.48,+.16,+.20,+.28,-.82,-.34,+.05),
 "noop":(0.,0.,0.,0.,0.,0.,0.)}
def utility_after(z,op):
 # Independent evaluator: lower deficits/pressure/load/interference are better; plasticity is valuable only when gap exists.
 e=EFFECTS[op];x=[max(0.,min(1.,a+b)) for a,b in zip(z,e)]
 gap,plastic,red,load,archive,pressure,interf=x
 value=-(1.4*gap+1.0*red+1.0*load+1.15*pressure+1.1*interf)
 value+=.55*plastic*gap
 # Archive relevance is opportunity, not intrinsic cost; unused relevant archive carries a small opportunity penalty.
 value-=.45*archive
 # Complexity/intervention penalty prevents gratuitous mutation and makes NOOP a real competitor.
 value-= {"create":.34,"modify":.18,"combine":.30,"freeze":.10,"prune":.12,"restore":.16,"noop":0.}[op]
 return value
def oracle(z):return max(OPS,key=lambda op:utility_after(z,op))
class FrozenPi:
 """Fixed linear contextual policy. Parameters are preregistered, never updated during benchmark."""
 def __init__(self):
  self.w={
   "create":(1.05,.62,-.32,-.18,-.20,.58,.05),
   "modify":(.35,.28,.05,.02,.00,.25,.42),
   "combine":(.48,.22,.38,-.08,.05,.35,-.05),
   "freeze":(-.15,-.30,.18,.35,-.05,-.12,.30),
   "prune":(-.18,-.20,.92,.82,-.10,-.12,.52),
   "restore":(.42,.28,-.08,-.05,1.05,.42,.02),
   "noop":(-.28,-.22,-.28,-.28,-.25,-.35,-.25)}
 def choose(self,z,rng):
  # Soft stochastic choice: no hand-coded state branches.
  scores={o:sum(a*b for a,b in zip(self.w[o],z)) for o in OPS}
  temp=.22;mx=max(scores.values());ex={o:math.exp((v-mx)/temp) for o,v in scores.items()};tot=sum(ex.values());u=rng.random()*tot
  acc=0.
  for o in OPS:
   acc+=ex[o]
   if u<=acc:return o
  return OPS[-1]
def wilson(k,n,z=1.96):
 p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return c-h,c+h
def run(seeds=range(100)):
 pi=FrozenPi();rows={};passed=True
 for name,base in SCENARIOS.items():
  # Noise makes states overlap; oracle is computed on the realized state, not scenario name.
  hits=0;counts={o:0 for o in OPS};oracle_counts={o:0 for o in OPS}
  for seed in seeds:
   rng=random.Random(10000*list(SCENARIOS).index(name)+seed)
   z=tuple(max(0.,min(1.,v+rng.gauss(0,.055))) for v in base)
   best=oracle(z);choice=pi.choose(z,rng);hits+=choice==best;counts[choice]+=1;oracle_counts[best]+=1
  rate=hits/len(seeds);lo,hi=wilson(hits,len(seeds));ok=rate>=.60 and lo>.50;passed &= ok
  rows[name]={"n":len(seeds),"optimal_choice_rate":rate,"ci95":[lo,hi],"choice_counts":counts,"oracle_counts":oracle_counts,"passed":ok}
 out={"schema":"symcore.bidirectional-choice.v1","operations":list(OPS),"features":list(FEATURES),
      "threshold":.60,"ci95_lower_required":.50,"seeds_per_state":len(seeds),"states":rows,"passed":bool(passed)}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/bidirectional_choice_report.json").write_text(json.dumps(out,indent=2))
 print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
