#!/usr/bin/env python3
"""Cross-vocabulary semantic transfer: train universal state->effect policy on source experience, test on Host3 local mechanisms."""
import json,math,random,statistics
from pathlib import Path
from acquisition_state import AcquisitionState
from intervention_effects import Effect,INTENTS,cosine
from effect_adapter import EffectAdapter
from semantic_growth_policy import SemanticGrowthPolicy
from host3_bandit import env_probs
SOURCE_EFFECTS={"source_plastic":Effect(plasticity=.55,exploration=.45,stability=.25,representation=.35),
                "source_stable":Effect(stability=.60,plasticity=.20,representation=.55,exploration=-.15),
                "source_reset":Effect(reset=.55,plasticity=.25,representation=.45,stability=.20)}
TARGET_EFFECTS={"fast":Effect(plasticity=1,exploration=.8,stability=-.4),
                "slow":Effect(stability=.9,plasticity=.1,exploration=.1),
                "reset":Effect(reset=1,plasticity=.6,exploration=.2)}
def state(trials,success,qs,vel,transfer):
 ent=-sum(max(1e-9,q)*math.log(max(1e-9,q)) for q in qs)/len(qs);s=sorted(qs,reverse=True);margin=s[0]-s[1]
 return AcquisitionState(trials,success,ent,margin,vel,transfer)
def source_policy(seed):
 rng=random.Random(seed);p=SemanticGrowthPolicy(3)
 # source experience generated in universal acquisition-state space; local names never enter policy.
 for _ in range(900):
  success=rng.random();qs=[rng.random() for _ in range(3)];vel=rng.uniform(-.3,.3);transfer=rng.uniform(-.2,.3);s=state(rng.randrange(1,200),success,qs,vel,transfer)
  # causal source utility: low success/negative velocity benefits plasticity/reset; stable high-success benefits consolidation.
  best="adapt_fast" if success<.45 and vel<=0 else "reset_state" if transfer<-.05 else "consolidate"
  for intent in ("adapt_fast","consolidate","reset_state"):
   u=(1. if intent==best else -.2)+rng.gauss(0,.15);p.observe(s,intent,u)
 return p
def run_arm(seed,semantic,T=720):
 rng=random.Random(seed);p=source_policy(seed+10000) if semantic else None;adapter=EffectAdapter(TARGET_EFFECTS);q=[.5]*3;n=[0]*3;reg=0.;bad=0;recent=[];prev=.0
 for t in range(T):
  probs=env_probs(t,seed);succ=statistics.fmean(recent[-20:]) if recent else .5;vel=succ-(statistics.fmean(recent[-40:-20]) if len(recent)>=40 else succ);s=state(t+1,succ,q,vel,0.)
  if semantic:intent=p.choose(s);op=adapter.decode(intent)
  else:op=("fast","slow","reset")[rng.randrange(3)]
  eps={"fast":.28,"slow":.08,"reset":.16}[op]
  arm=rng.randrange(3) if rng.random()<eps else max(range(3),key=q.__getitem__);reward=int(rng.random()<probs[arm]);recent.append(reward);step=max(probs)-probs[arm];reg+=step;bad+=step>.35
  lr={"fast":.22,"slow":.06,"reset":.12}[op];q[arm]=(1-lr)*q[arm]+lr*reward
 return reg,bad/T
def ci(x):
 m=statistics.fmean(x);se=statistics.stdev(x)/(len(x)**.5);return [m-1.96*se,m+1.96*se]
def run():
 seeds=range(300);base=[run_arm(s,False) for s in seeds];sem=[run_arm(s,True) for s in seeds];imp=[a[0]-b[0] for a,b in zip(base,sem)];risk=[b[1]-a[1] for a,b in zip(base,sem)]
 out={"schema":"symcore.semantic_transfer.v1","seeds":300,"source_target_local_names_disjoint":True,
      "regret_improvement":statistics.fmean(imp),"regret_ci95":ci(imp),"risk_delta":statistics.fmean(risk),"risk_ci95":ci(risk),
      "passed":ci(imp)[0]>0 and ci(risk)[1]<=0}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/semantic_transfer_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
