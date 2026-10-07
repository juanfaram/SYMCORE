#!/usr/bin/env python3
"""Non-temporal tabular classification host: same stream, paired online control vs SYMCORE feature-specialist router."""
import csv,json,math,random,statistics,urllib.request
from pathlib import Path
from host_advantage import paired_advantage
URL="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
class OnlineLogit:
 def __init__(self,idx,lr=.08):self.idx=idx;self.w=[0.]*(len(idx)+1);self.lr=lr
 def p(self,x):
  z=self.w[0]+sum(self.w[i+1]*x[j] for i,j in enumerate(self.idx));z=max(-20,min(20,z));return 1/(1+math.exp(-z))
 def step(self,x,y):
  p=self.p(x);e=p-y;self.w[0]-=self.lr*e
  for i,j in enumerate(self.idx):self.w[i+1]-=self.lr*e*x[j]
  return p
def run():
 p=Path("data/pima.csv");p.parent.mkdir(exist_ok=True)
 if not p.exists():urllib.request.urlretrieve(URL,p)
 rows=[]
 with p.open() as f:
  for r in csv.reader(f):
   vals=list(map(float,r));x=vals[:-1];y=int(vals[-1]);rows.append((x,y))
 # normalize causally with fixed public-domain scale constants to avoid future leakage
 scales=[20,200,130,100,900,70,1.0,100];rows=[([v/s for v,s in zip(x,scales)],y) for x,y in rows]
 control=OnlineLogit(tuple(range(8)),.06)
 experts=[OnlineLogit((0,1,5,7),.06),OnlineLogit((1,2,3,4,5),.06),OnlineLogit(tuple(range(8)),.03)]
 losses={i:[] for i in range(len(experts))};cl=[];sl=[]
 for x,y in rows:
  cp=control.step(x,y);cl.append(abs(cp-y))
  preds=[e.p(x) for e in experts];mature={i:statistics.fmean(v[-80:]) for i,v in losses.items() if len(v)>=30};pick=min(mature,key=mature.get) if mature else 2
  sl.append(abs(preds[pick]-y))
  for i,e in enumerate(experts):losses[i].append(abs(preds[i]-y));e.step(x,y)
 ev=paired_advantage(cl[100:],sl[100:])
 out={"paradigm":"tabular_binary_classification","rows":len(rows),"paired":True,"evidence":ev}
 Path("artifacts/tabular_host_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
