#!/usr/bin/env python3
"""Real-data paired host trial: same causal stream, control vs SYMCORE-assisted host."""
import csv,json,math,statistics
from pathlib import Path
from seasonal import SeasonalMemory
from main import ensure

class ControlHost:
 def __init__(self):self.model=SeasonalMemory(("hr",),8)
 def step(self,r,y):
  p=self.model.predict(r);self.model.learn(r,y);return abs(p-y)

class SymcoreHost:
 def __init__(self):
  self.experts={"hour":SeasonalMemory(("hr",),8),"work":SeasonalMemory(("hr","workingday"),8),
                "weekday":SeasonalMemory(("hr","weekday"),8),"weather":SeasonalMemory(("hr","weathersit"),8)}
  self.loss={k:[] for k in self.experts};self.promotions=0
 def step(self,r,y):
  preds={k:m.predict(r) for k,m in self.experts.items()}
  # strictly causal: choose using past losses only, predict, then reveal y and update.
  mature={k:statistics.fmean(v[-256:]) for k,v in self.loss.items() if len(v)>=64}
  chosen=min(mature,key=mature.get) if mature else "hour"
  p=preds[chosen]
  for k,m in self.experts.items():
   self.loss[k].append(abs(preds[k]-y));m.learn(r,y)
  return abs(p-y)

def run():
 path=Path("data/hour.csv");ensure(path);control=ControlHost();sym=SymcoreHost()
 ec=[];es=[];wins=[]
 with path.open() as f:
  for i,r in enumerate(csv.DictReader(f),1):
   y=float(r["cnt"]);ec.append(control.step(r,y));es.append(sym.step(r,y))
   if i%1024==0 and i>=2048:
    c=statistics.fmean(ec[-1024:]);s=statistics.fmean(es[-1024:])
    wins.append({"at":i,"control_mae":round(c,3),"symcore_mae":round(s,3),
                 "A":round(math.log(c/max(s,1e-9)),6)})
 c=statistics.fmean(ec[-2048:]);s=statistics.fmean(es[-2048:])
 dif=[a-b for a,b in zip(ec[-2048:],es[-2048:])]
 mean=statistics.fmean(dif);se=statistics.stdev(dif)/math.sqrt(len(dif))
 out={"dataset":"UCI Bike Sharing hour.csv","rows":len(ec),"paired":True,
      "control_mae_last2048":round(c,3),"symcore_mae_last2048":round(s,3),
      "A":round(math.log(c/max(s,1e-9)),6),"paired_gain":round(mean,3),
      "gain_ci95":[round(mean-1.96*se,3),round(mean+1.96*se,3)],
      "windows":wins,"positive_A_fraction":round(sum(x["A"]>0 for x in wins)/max(1,len(wins)),4)}
 Path("artifacts/real_host_control_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
