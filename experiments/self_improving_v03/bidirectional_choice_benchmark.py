#!/usr/bin/env python3
"""460 benchmark v2: frozen Pi chooses operations; Frontier-v3 geometry defines ex-post optimality."""
from __future__ import annotations
import json,math,random
from learned_evolution_pi import LearnedEvolutionPi
from pathlib import Path
OPS=("create","modify","combine","freeze","prune","restore","noop")
FEATURES=("capacity_gap","plasticity","redundancy","load","archive_relevance","frontier_pressure","interference")
SCENARIOS={
 "redundant_loaded":(.20,.45,.82,.88,.10,.25,.40),
 "capacity_plastic":(.84,.86,.18,.35,.05,.72,.28),
 "archive_relevant":(.52,.55,.30,.48,.91,.61,.32),
 "stable_unpressured":(.08,.28,.08,.10,.05,.06,.08)}
# Effects on Frontier axes (Q,A,R,E,L), not on scenario labels.
DELTA={
 "create":(.16,.12,-.03,-.18,.10),"modify":(.07,.06,.01,-.04,.05),"combine":(.10,.08,.04,-.10,.08),
 "freeze":(-.02,-.04,.07,.10,.01),"prune":(-.03,-.02,.03,.22,.07),"restore":(.12,.10,.09,-.08,.13),"noop":(0.,0.,0.,0.,0.)}
def baseline(z):
 gap,plastic,red,load,archive,pressure,interf=z
 return (.78-.32*gap-.12*interf,.62+.20*plastic-.20*pressure,.82-.22*interf-.10*gap,.82-.40*load-.22*red,.60+.22*plastic-.18*pressure-.12*interf)
def outcome(z,op):
 b=baseline(z);d=list(DELTA[op]);gap,plastic,red,load,archive,pressure,interf=z
 # Context modulates generic effects; no state->operation branch exists.
 if op=="create":d[0]*=gap;d[1]*=plastic;d[4]*=gap*plastic
 if op=="modify":d[0]*=(gap+interf)/2;d[1]*=plastic
 if op=="combine":d[0]*=gap;d[2]*=(1-interf)
 if op=="freeze":d[2]*=(interf+pressure)/2;d[3]*=load
 if op=="prune":d[3]*=(red+load)/2;d[4]*=red;d[0]*=(.35+red)
 if op=="restore":d[0]*=archive;d[1]*=archive;d[2]*=archive;d[4]*=archive
 return tuple(max(0.,min(1.,x+y)) for x,y in zip(b,d))
def pareto_gain(z,op):
 b=baseline(z);x=outcome(z,op);tol=.012
 # intolerable regression budget mirrors Frontier logic; otherwise count robust axis expansions.
 if any(a<c-.05 for a,c in zip(x,b)):return -999.
 gains=[a-c for a,c in zip(x,b)]
 return sum(max(0.,g-tol) for g in gains)-.35*sum(max(0.,-g-tol) for g in gains)
def oracle(z):
 vals={o:pareto_gain(z,o) for o in OPS};best=max(vals.values())
 # NOOP wins whenever no intervention clears a meaningful evolutionary margin.
 if best<.025:return "noop"
 return max(vals,key=vals.get)
def train_pi():
 # Separate experience stream: broad functional states, all operations evaluated by the frozen Frontier-v3-aligned outcome model.
 # Training never sees scenario names, test seeds, or oracle labels.
 pi=LearnedEvolutionPi(k=48,temperature=.035,exploration=.04);rng=random.Random(460031)
 for _ in range(2400):
  z=tuple(rng.random() for _ in FEATURES)
  for op in OPS:pi.observe(z,op,pareto_gain(z,op))
 pi.fit()
 return pi.frozen()
def wilson(k,n,z=1.96):
 p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return c-h,c+h
def run(seeds=range(100)):
 pi=train_pi();rows={};passed=True
 for idx,(name,base) in enumerate(SCENARIOS.items()):
  hits=0;counts={o:0 for o in OPS};oracle_counts={o:0 for o in OPS}
  for seed in seeds:
   rng=random.Random(idx*10000+seed);z=tuple(max(0.,min(1.,v+rng.gauss(0,.055))) for v in base)
   best=oracle(z);choice=pi.choose(z,rng);hits+=choice==best;counts[choice]+=1;oracle_counts[best]+=1
  rate=hits/len(seeds);lo,hi=wilson(hits,len(seeds));ok=rate>=.60 and lo>.50;passed &= ok
  rows[name]={"n":len(seeds),"optimal_choice_rate":rate,"ci95":[lo,hi],"choice_counts":counts,"oracle_counts":oracle_counts,"passed":ok}
 out={"schema":"symcore.bidirectional-choice.v2","oracle":"frontier-v3-aligned","policy":"learned-operation-specific-context-geometry","training_states":2400,"threshold":.60,"ci95_lower_required":.50,"seeds_per_state":len(seeds),"states":rows,"passed":bool(passed)}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/bidirectional_choice_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
