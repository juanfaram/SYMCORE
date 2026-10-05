#!/usr/bin/env python3
"""Host 3: non-stationary bandit. Test whether history+online growth policy beats online-only.
Gap is causally observable only as drift/stable from a residual change detector."""
import json,math,random,statistics
from pathlib import Path
from operator_policy import OperatorPolicy
from intervention_effects import Effect,INTENTS
from effect_memory import EffectMemory
OPS=("fast","slow","reset")
LOCAL_EFFECTS={"fast":Effect(plasticity=1.,exploration=.8,stability=-.25),"slow":Effect(stability=1.,plasticity=.15),"reset":Effect(reset=1.,plasticity=.65)}
# Historical evidence comes from prior acquisition behavior, not hidden Host3 regimes.
HIST=[("drift","adapt_fast",1),("drift","reset_state",1),("drift","consolidate",0),("stable","consolidate",1),("stable","balanced_adapt",1),("stable","reset_state",0)]
class Detector:
 def __init__(self):self.e=[] 
 def gap(self):
  if len(self.e)<24:return "stable"
  a=statistics.fmean(self.e[-8:]);b=statistics.fmean(self.e[-24:-8]);sd=statistics.pstdev(self.e[-24:]) or 1
  return "drift" if a>b+.6*sd else "stable"
 def observe(self,e):self.e.append(float(e))
def env_probs(t,seed):
 # unseen randomized regime boundaries and best arms
 r=random.Random(seed+999);bounds=[0,180+r.randrange(-25,26),360+r.randrange(-25,26),540+r.randrange(-25,26),720]
 best=[r.randrange(3) for _ in range(4)];phase=max(i for i,b in enumerate(bounds[:-1]) if t>=b)
 # Regimes differ in both best arm and sharpness: shock phases reward fast adaptation,
 # consolidation phases punish exploration, recovery phases reward reset/relearning.
 sharp=(.86,.58,.82,.56)[phase];floor=(.12,.36,.16,.38)[phase]
 ps=[floor,floor,floor];ps[best[phase]]=sharp;return ps
def simulate(seed,adaptive,h=0.,T=720):
 rng=random.Random(seed);p=OperatorPolicy(operators=OPS,history_strength=h);det=Detector();mem=EffectMemory(prior=max(.25,1./max(h,.1)))
 if adaptive:
  for g,intent,s in HIST:
   # History is stored in the shared effect space; no Host3 operator name appears here.
   for _ in range(max(1,int(round(4*h)))):mem.observe(g,INTENTS[intent],s)
 q=[.5]*3;n=[0]*3;regret=0.;viol=0
 for t in range(T):
  gap=det.gap()
  if adaptive:
   ranked=mem.rank(gap,LOCAL_EFFECTS);eps_transfer=.15
   op=rng.choice(OPS) if rng.random()<eps_transfer else ranked[0]
  else:op=p.choose(gap,rng)
  # growth operator controls adaptation dynamics, not action directly
  if op=="reset" and gap=="drift":q=[.5]*3;n=[0]*3
  eps=.28 if op=="fast" else .08 if op=="slow" else .16
  arm=rng.randrange(3) if rng.random()<eps else max(range(3),key=q.__getitem__)
  probs=env_probs(t,seed);reward=int(rng.random()<probs[arm]);best=max(probs);step_regret=best-probs[arm];regret+=step_regret;viol+=step_regret>.35
  n[arm]+=1;lr=.22 if op=="fast" else .06 if op=="slow" else .12;q[arm]=(1-lr)*q[arm]+lr*reward
  det.observe(abs(reward-q[arm]));p.update(gap,op,reward)
  if adaptive:mem.observe(gap,LOCAL_EFFECTS[op],reward)
 return regret,viol/T
def ci(ds):
 m=statistics.fmean(ds);se=statistics.stdev(ds)/(len(ds)**.5);return [m-1.96*se,m+1.96*se]
def run():
 seeds=range(200);strengths=(.10,.25,.50,1.,2.,4.);base=[simulate(s,False) for s in seeds];curve=[]
 for h in strengths:
  xs=[simulate(s,True,h) for s in seeds];improve=[a[0]-b[0] for a,b in zip(base,xs)];risk=[b[1]-a[1] for a,b in zip(base,xs)]
  curve.append({"h":h,"regret_improvement":statistics.fmean(improve),"regret_ci95":ci(improve),
                "risk_delta":statistics.fmean(risk),"risk_delta_ci95":ci(risk)})
 positive=[x for x in curve if x["regret_ci95"][0]>0 and x["risk_delta_ci95"][1]<=0]
 out={"schema":"symcore.host3.v1","seeds":len(list(seeds)),"gap_definition":"causal detector: drift/stable","unseen_regimes":True,
      "curve":curve,"positive_safe_strengths":len(positive),"required":3,"passed":len(positive)>=3}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/host3_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
