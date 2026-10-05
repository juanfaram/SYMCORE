#!/usr/bin/env python3
"""Three-arm real-data trial: baseline, equally-online host, and SYMCORE-assisted host."""
import csv,json,math,statistics
from pathlib import Path
from seasonal import SeasonalMemory
from main import ensure
from intervention_controller import InterventionController
class Host:
 def __init__(self,keys):self.model=SeasonalMemory(keys,8)
 def step(self,r,y):
  p=self.model.predict(r);self.model.learn(r,y);return abs(p-y)
class SymcoreHost:
 def __init__(self):
  self.experts={"hour":SeasonalMemory(("hr",),8),"work":SeasonalMemory(("hr","workingday"),8),
                "weekday":SeasonalMemory(("hr","weekday"),8),"weather":SeasonalMemory(("hr","weathersit"),8)}
  self.loss={k:[] for k in self.experts}
 def step(self,r,y):
  preds={k:m.predict(r) for k,m in self.experts.items()}
  mature={k:statistics.fmean(v[-256:]) for k,v in self.loss.items() if len(v)>=64}
  chosen=min(mature,key=mature.get) if mature else "hour";p=preds[chosen]
  for k,m in self.experts.items():self.loss[k].append(abs(preds[k]-y));m.learn(r,y)
  return abs(p-y)
def stats(a,b):
 ma=statistics.fmean(a);mb=statistics.fmean(b);d=[x-y for x,y in zip(a,b)]
 mean=statistics.fmean(d);se=statistics.stdev(d)/math.sqrt(len(d))
 return {"reference_mae":round(ma,3),"symcore_mae":round(mb,3),"A":round(math.log(ma/max(mb,1e-9)),6),
         "gain":round(mean,3),"gain_ci95":[round(mean-1.96*se,3),round(mean+1.96*se,3)]}
def run():
 path=Path("data/hour.csv");ensure(path)
 baseline=Host(("hr",));online=Host(("hr","workingday"));sym=SymcoreHost();selective=SymcoreHost();ctl=InterventionController()
 eb=[];eo=[];es=[];ex=[];windows=[];drift_flags=[];active_flags=[];recent=[];active_count=0
 with path.open() as f:
  for i,r in enumerate(csv.DictReader(f),1):
   y=float(r["cnt"])
   # Intervention decision is made before observing this outcome.
   active=ctl.decide()
   be=baseline.step(r,y);oe=online.step(r,y);se=sym.step(r,y)
   xe=selective.step(r,y) if active else oe
   eb.append(be);eo.append(oe);es.append(se);ex.append(xe);active_flags.append(active);active_count+=int(active)
   ctl.observe(oe)
   # Drift label is computed from online-control error only, before using SYMCORE advantage.
   hist=recent[-256:];mu=statistics.fmean(hist) if hist else oe;sd=statistics.stdev(hist) if len(hist)>1 else 0.
   drift_flags.append(len(hist)>=128 and oe>mu+1.5*sd);recent.append(oe)
   if i%1024==0 and i>=2048:
    windows.append({"at":i,"vs_baseline":stats(eb[-1024:],es[-1024:]),"vs_online":stats(eo[-1024:],es[-1024:])})
 drift_idx=[i for i,x in enumerate(drift_flags) if x];stable_idx=[i for i,x in enumerate(drift_flags) if not x]
 def subset(ix):return stats([eo[i] for i in ix],[es[i] for i in ix]) if len(ix)>1 else None
 active_idx=[i for i,x in enumerate(active_flags) if x];inactive_idx=[i for i,x in enumerate(active_flags) if not x]
 drift_active=sum(active_flags[i] for i in drift_idx);stable_active=sum(active_flags[i] for i in stable_idx)
 discrimination={"active_rate_drift":round(drift_active/max(1,len(drift_idx)),4),
                 "active_rate_stable":round(stable_active/max(1,len(stable_idx)),4),
                 "precision_for_drift":round(drift_active/max(1,len(active_idx)),4),
                 "recall_of_drift":round(drift_active/max(1,len(drift_idx)),4),
                 "motor_advantage_when_active":subset(active_idx),
                 "motor_advantage_when_inactive":subset(inactive_idx)}
 out={"dataset":"UCI Bike Sharing hour.csv","rows":len(eb),"paired":True,
      "drift_conditioned":{"drift_n":len(drift_idx),"stable_n":len(stable_idx),"drift":subset(drift_idx),"stable":subset(stable_idx)},
      "vs_baseline":stats(eb[-2048:],es[-2048:]),"vs_online_control":stats(eo[-2048:],es[-2048:]),
      "selective_intervention":{"active_fraction":round(active_count/len(eo),4),"vs_online":stats(eo[-2048:],ex[-2048:]),"vs_always_on":stats(es[-2048:],ex[-2048:]),"discrimination":discrimination},
      "positive_A_vs_online_fraction":round(sum(w["vs_online"]["A"]>0 for w in windows)/len(windows),4),
      "windows":windows}
 Path("artifacts/real_host_control_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
