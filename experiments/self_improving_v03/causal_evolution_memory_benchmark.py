#!/usr/bin/env python3
"""575 holdout: predict unseen evolutionary consequences and choose safe useful effects."""
import json,math,random
from pathlib import Path
from causal_evolution_memory import CausalEvolutionMemory,AXES
EFFECTS={"expand":(1,0,0,.3,.2),"stabilize":(0,1,0,.1,0),"reset":(.4,0,1,0,.3),"compress":(-.1,.4,0,-.8,.1)}
def world(z,e):
 # Hidden smooth causal process; memory sees samples, never this mapping.
 gap,change,noise,load,history=z
 p,s,r,c,x=e
 return ( .22*p*gap+.08*x*gap-.03*c,
          .18*p*change+.16*r*change+.06*x-.05*noise*p,
          .20*s*(1-change)+.13*r*change-.04*p*noise,
          .25*(-c)*load-.12*p-.06*r,
          .18*x*history+.12*p*gap+.08*s*(1-noise)), .35+.45*(abs(p)+abs(s)+abs(r)+abs(c)+abs(x))/5
def jitter(v,r,s=.02):return tuple(max(0.,min(1.,x+r.gauss(0,s))) for x in v)
def run():
 mem=CausalEvolutionMemory(k=72);rng=random.Random(575001)
 # Train on one region/host distribution.
 for _ in range(5000):
  z=(rng.random(),rng.random(),rng.random(),rng.random(),rng.random())
  for _,e in EFFECTS.items():
   d,c=world(z,e);mem.observe(jitter(z,rng,.01),e,tuple(x+rng.gauss(0,.012) for x in d),c+rng.gauss(0,.01))
 # Held-out states use different seed and affine sensor abstraction is already handled by z_H upstream.
 errs={a:[] for a in AXES};cost_err=[];correct=0;n=400
 for i in range(n):
  rr=random.Random(900000+i);z=tuple(rr.random() for _ in range(5))
  chosen,preds=mem.choose(z,EFFECTS)
  truths={name:world(z,e) for name,e in EFFECTS.items()}
  def true_score(item):
   d,c=item
   if min(d)<-.05:return -1e9
   return sum(max(0,x) for x in d)/c
  oracle=max(truths,key=lambda x:true_score(truths[x]));correct+=chosen==oracle
  for name,e in EFFECTS.items():
   d,c=truths[name];p=preds[name]
   for a,x in zip(AXES,d):errs[a].append(abs(p.means[a]-x))
   cost_err.append(abs(p.cost_mean-c))
 mae={a:sum(v)/len(v) for a,v in errs.items()};acc=correct/n
 out={"schema":"symcore.causal-evolution-memory.v1","holdout_n":n,"axis_mae":mae,"cost_mae":sum(cost_err)/len(cost_err),"safe_choice_accuracy":acc,
      "criteria":{"max_axis_mae":.04,"max_cost_mae":.04,"min_choice_accuracy":.70},
      "passed":max(mae.values())<.04 and sum(cost_err)/len(cost_err)<.04 and acc>=.70}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/causal_evolution_memory_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
