#!/usr/bin/env python3
"""Useful diversity gate for tabular specialists: distinct errors + unique wins + router gain."""
import csv,json,math,statistics,urllib.request
from pathlib import Path
from tabular_capability_factory import TabularCapabilityFactory
from host_advantage import paired_advantage
class Logit:
 def __init__(self,idx,lr=.05):self.idx=tuple(idx);self.w=[0.]*(len(self.idx)+1);self.lr=lr
 def pred(self,x):
  z=self.w[0]+sum(self.w[i+1]*x[j] for i,j in enumerate(self.idx));z=max(-20,min(20,z));return 1/(1+math.exp(-z))
 def learn(self,x,y):
  p=self.pred(x);e=p-y;self.w[0]-=self.lr*e
  for i,j in enumerate(self.idx):self.w[i+1]-=self.lr*e*x[j]
def run():
 p=Path("data/pima.csv");p.parent.mkdir(exist_ok=True)
 if not p.exists():urllib.request.urlretrieve("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",p)
 raw=[list(map(float,r)) for r in csv.reader(p.open())];sc=[20,200,130,100,900,70,1,100]
 rows=[([v/s for v,s in zip(r[:-1],sc)],int(r[-1])) for r in raw]
 # feature signal from first half only: causal factory input
 half=len(rows)//2;effects={}
 for j in range(8):
  pos=[x[j] for x,y in rows[:half] if y];neg=[x[j] for x,y in rows[:half] if not y]
  effects[j]=abs(statistics.fmean(pos)-statistics.fmean(neg))
 specs=TabularCapabilityFactory().propose(8,effects,4,6);experts=[Logit(s["features"]) for s in specs];loss=[[] for _ in experts];router=[];anchor=[];wins=[0]*len(experts)
 for x,y in rows[half:]:
  ps=[e.pred(x) for e in experts];errs=[abs(p-y) for p in ps];best=min(range(len(errs)),key=errs.__getitem__);wins[best]+=1
  mature={i:statistics.fmean(v[-80:]) for i,v in enumerate(loss) if len(v)>=30};pick=min(mature,key=mature.get) if mature else 0
  router.append(errs[pick]);anchor.append(errs[0])
  for i,e in enumerate(experts):loss[i].append(errs[i]);e.learn(x,y)
 useful=sum(w>10 for w in wins)>=2;ev=paired_advantage(anchor,router)
 out={"specs":[s["features"] for s in specs],"unique_wins":wins,"useful_diversity":useful,"router_vs_best_signal_anchor":ev,
      "gate_pass":useful and ev["passed"]}
 Path("artifacts/tabular_diversity_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
