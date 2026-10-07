#!/usr/bin/env python3
"""Paired contextual-router evaluation with frozen specialist definitions."""
import csv,json,math,statistics,urllib.request
from pathlib import Path
from tabular_capability_factory import TabularCapabilityFactory
from context_router import ContextRouter
from host_advantage import paired_advantage
from tabular_diversity_benchmark import Logit
def run():
 p=Path("data/pima.csv");p.parent.mkdir(exist_ok=True)
 if not p.exists():urllib.request.urlretrieve("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",p)
 raw=[list(map(float,r)) for r in csv.reader(p.open())];sc=[20,200,130,100,900,70,1,100];rows=[([v/s for v,s in zip(r[:-1],sc)],int(r[-1])) for r in raw];half=len(rows)//2
 effects={}
 for j in range(8):
  pos=[x[j] for x,y in rows[:half] if y];neg=[x[j] for x,y in rows[:half] if not y];effects[j]=abs(statistics.fmean(pos)-statistics.fmean(neg))
 specs=TabularCapabilityFactory().propose(8,effects,4,6);experts=[Logit(s["features"]) for s in specs];router=ContextRouter(len(experts));flat=[];ctx=[]
 losses=[[] for _ in experts]
 for x,y in rows[half:]:
  ps=[e.pred(x) for e in experts];errs=[abs(p-y) for p in ps];m={i:statistics.fmean(v[-80:]) for i,v in enumerate(losses) if len(v)>=30};fp=min(m,key=m.get) if m else 0;cp=router.choose(x)
  flat.append(errs[fp]);ctx.append(errs[cp]);router.feedback(x,errs)
  for i,e in enumerate(experts):losses[i].append(errs[i]);e.learn(x,y)
 ev=paired_advantage(flat,ctx);out={"contextual_vs_flat":ev,"passed":ev["passed"]}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/contextual_router_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
