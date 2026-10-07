#!/usr/bin/env python3
"""Diagnose temporal variance of selective concentration R without changing controller."""
import csv,json,statistics
from pathlib import Path
from main import ensure
from real_host_control import Host,SymcoreHost
from intervention_controller import InterventionController

def R_for(delta,g):
 rate=sum(g)/max(1,len(g));total=sum(delta)
 return sum(x*y for x,y in zip(delta,g))/max(1e-12,rate*total) if rate>0 and total>0 else None
def episodes(delta):
 out=[];n=0
 for x in delta:
  if x>0:n+=1
  elif n:out.append(n);n=0
 if n:out.append(n)
 return out
def diagnose(block):
 path=Path("data/hour.csv");ensure(path);rows=list(csv.DictReader(path.open()));split=int(len(rows)*.60)
 online=Host(("hr","workingday"));sym=SymcoreHost();ctl=InterventionController();d=[];g=[]
 for r in rows:
  active=ctl.decide();y=float(r["cnt"]);oe=online.step(r,y);se=sym.step(r,y);d.append(oe-se);g.append(int(active));ctl.observe(oe)
 d=d[split:];g=g[split:];blocks=[(d[i:i+block],g[i:i+block]) for i in range(0,len(d),block)]
 vals=[sum(x*y for x,y in zip(db,gb)) for db,gb in blocks];positive=[max(0,x) for x in vals]
 totalp=sum(positive);ordered=sorted(positive,reverse=True);top=max(1,(len(ordered)+19)//20)
 top_share=sum(ordered[:top])/totalp if totalp else 0
 # Gini on nonnegative captured-value contribution.
 xs=sorted(positive);n=len(xs);gini=(2*sum((i+1)*x for i,x in enumerate(xs))/(n*sum(xs))-(n+1)/n) if sum(xs)>0 else 0
 loo=[]
 for k in range(len(blocks)):
  dd=[];gg=[]
  for j,(db,gb) in enumerate(blocks):
   if j!=k:dd+=db;gg+=gb
  q=R_for(dd,gg)
  if q is not None:loo.append(q)
 ep=episodes(d)
 return {"block":block,"n_blocks":len(blocks),"R":R_for(d,g),"top5pct_positive_captured_share":top_share,
         "gini_positive_captured":gini,"loo_R_min":min(loo) if loo else None,"loo_R_max":max(loo) if loo else None,
         "positive_episode_count":len(ep),"episode_median":statistics.median(ep) if ep else 0,
         "episode_mean":statistics.fmean(ep) if ep else 0,"episode_max":max(ep) if ep else 0}
def run():
 out={"schema":"symcore.R_variance_diagnosis.v1","controller_frozen":True,
      "diagnostics":[diagnose(b) for b in (64,128,256)]}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/R_variance_diagnosis.json").write_text(json.dumps(out,indent=2))
 print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
