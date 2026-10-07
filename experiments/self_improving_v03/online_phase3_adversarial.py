#!/usr/bin/env python3
"""Phase 3 adversarial stream: preregistered robustness envelope for online-learning instrumentation."""
from __future__ import annotations
import json,math,random,statistics
from pathlib import Path
CHECKPOINTS=tuple(range(100,1201,100));SEEDS=tuple(range(40))
LEVELS={
 "regime_changes":(0,1,5,10),
 "feedback_corruption":(0.,.01,.05,.10),
 "rare_states":(0.,.01,.05,.10),
 "cost_variance":(0.,.10,.50,1.00)}
# Gate to unlock real host: clean + moderate severity must pass for every adversary.
REQUIRED={"regime_changes":5,"feedback_corruption":.05,"rare_states":.05,"cost_variance":.50}
def slope(rows,col):
 xs=[r[0] for r in rows];ys=[r[col] for r in rows];xm=statistics.fmean(xs);ym=statistics.fmean(ys)
 return sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/sum((x-xm)**2 for x in xs)
def q(v,p):
 a=sorted(v);return a[min(len(a)-1,max(0,int(p*(len(a)-1))))]
def one(seed,kind,level):
 r=random.Random(330000+seed+sum(map(ord,kind))*1000+int(level*1000 if isinstance(level,float) else level))
 rows=[];skill=0.;frozen_skill=0.;freeze_at=200
 change_points=set()
 if kind=="regime_changes" and level:
  change_points={round((i+1)*1200/(level+1)) for i in range(int(level))}
 for E in range(1,1201):
  if E in change_points:skill*=.72
  learn=.00042*math.exp(-E/1700)
  if kind=="feedback_corruption" and r.random()<level:learn*=-.65
  rare=(kind=="rare_states" and r.random()<level)
  if rare:learn*=.28
  skill+=learn
  if E==freeze_at:frozen_skill=skill
  if E in CHECKPOINTS:
   regime_penalty=.015*len([x for x in change_points if x<=E])
   A=skill-regime_penalty+r.gauss(0,.0015)
   live_frozen=skill-frozen_skill if E>freeze_at else 0.
   base_cost=.95-.00034*E
   if kind=="cost_variance":base_cost+=r.gauss(0,.06*level)
   C=max(.08,base_cost)
   rows.append((E,A,C,live_frozen))
 return rows
def evaluate(kind,level):
 sa=[];sc=[];rates=[]
 for seed in SEEDS:
  rows=one(seed,kind,level);sa.append(slope(rows,1));sc.append(slope(rows,2))
  post=[r[3] for r in rows if r[0]>200];rates.append(sum(x>0 for x in post)/len(post))
 lcb=q(sa,.025);ucb=q(sc,.975);live=statistics.fmean(rates)
 passed=lcb>0 and ucb<0 and live>=.80
 return {"dA_dE_mean":statistics.fmean(sa),"dA_dE_lcb95":lcb,"dCnew_dE_mean":statistics.fmean(sc),"dCnew_dE_ucb95":ucb,"live_gt_frozen_rate":live,"passed":passed}
def run():
 matrix={}
 for kind,levels in LEVELS.items():matrix[kind]={str(v):evaluate(kind,v) for v in levels}
 required_pass={k:matrix[k][str(v)]["passed"] for k,v in REQUIRED.items()}
 out={"schema":"symcore.online-phase3-adversarial.v1","instrument_only":True,"real_host_evidence":False,
 "seeds_per_condition":len(SEEDS),"checkpoints":len(CHECKPOINTS),"levels":LEVELS,"required_for_host_unlock":REQUIRED,
 "matrix":matrix,"required_pass":required_pass,"passed":all(required_pass.values())}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/online_phase3_adversarial_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
