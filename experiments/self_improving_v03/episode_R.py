#!/usr/bin/env python3
"""Episode-aware uncertainty for selective intervention. Controller and 60/40 holdout remain frozen."""
import csv,json,random,statistics
from pathlib import Path
from main import ensure
from real_host_control import Host,SymcoreHost
from intervention_controller import InterventionController

def episodes(flags,start):
 out=[];i=start
 while i<len(flags):
  if not flags[i]:i+=1;continue
  j=i+1
  while j<len(flags) and flags[j]:j+=1
  out.append((i,j));i=j
 return out

def bootstrap_episode_R(delta,g,start,reps=5000,seed=20261005):
 eps=episodes(g,start);rng=random.Random(seed);hold=list(range(start,len(delta)))
 inactive=[i for i in hold if not g[i]]
 vals=[]
 if not eps:return [],eps
 # Resample intervention episodes and inactive points in chunks matched to median episode duration.
 med=max(1,int(statistics.median(e-s for s,e in eps)))
 inactive_chunks=[inactive[i:i+med] for i in range(0,len(inactive),med)]
 units=[list(range(s,e)) for s,e in eps]+inactive_chunks
 for _ in range(reps):
  sample=[]
  for __ in units:sample.extend(rng.choice(units))
  sample=[i for i in sample if i<len(delta)]
  if not sample:continue
  rate=sum(g[i] for i in sample)/len(sample);total=sum(delta[i] for i in sample)
  captured=sum(delta[i]*g[i] for i in sample)
  if rate>0 and total>0:vals.append(captured/(rate*total))
 return sorted(vals),eps

def run():
 p=Path("data/hour.csv");ensure(p);rows=list(csv.DictReader(p.open()));split=int(len(rows)*.60)
 online=Host(("hr","workingday"));motor=SymcoreHost();ctl=InterventionController();d=[];g=[]
 for r in rows:
  y=float(r["cnt"]);active=ctl.decide();oe=online.step(r,y);se=motor.step(r,y);d.append(oe-se);g.append(int(active));ctl.observe(oe)
 eps=episodes(g,split);lens=[e-s for s,e in eps];vals,_=bootstrap_episode_R(d,g,split)
 rate=sum(g[split:])/(len(g)-split);total=sum(d[split:]);capt=sum(d[i]*g[i] for i in range(split,len(g)))
 R=capt/(rate*total) if rate and total>0 else None
 ci=[vals[int(.025*len(vals))],vals[int(.975*len(vals))-1]] if vals else None
 out={"split":.60,"controller_frozen":True,"episodes":len(eps),"episode_lengths":lens,
      "episode_mean":statistics.fmean(lens) if lens else 0,"episode_median":statistics.median(lens) if lens else 0,
      "episode_p95":sorted(lens)[max(0,int(.95*len(lens))-1)] if lens else 0,"old_block":256,
      "R":R,"R_episode_bootstrap_ci95":ci,"confirmed_R":ci is not None and ci[0]>1}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/episode_R_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
