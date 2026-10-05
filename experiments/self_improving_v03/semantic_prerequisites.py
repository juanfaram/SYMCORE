#!/usr/bin/env python3
"""Prerequisite audit for semantic transfer: nontrivial geometry, measurable acquisition state, target niches."""
import json,math,random,statistics
from pathlib import Path
from intervention_effects import cosine,INTENTS
from semantic_transfer_benchmark import SOURCE_EFFECTS,TARGET_EFFECTS
from host3_bandit import env_probs
DIMS=("trials","success_rate","policy_entropy","posterior_margin","recent_learning_velocity","historical_transfer")
def op_regret(seed,op,start,end):
 rng=random.Random(seed);q=[.5]*3;reg=0.
 for t in range(end):
  ps=env_probs(t,seed);eps={"fast":.28,"slow":.08,"reset":.16}[op]
  arm=rng.randrange(3) if rng.random()<eps else max(range(3),key=q.__getitem__);reward=int(rng.random()<ps[arm]);step=max(ps)-ps[arm]
  if t>=start:reg+=step
  lr={"fast":.22,"slow":.06,"reset":.12}[op];q[arm]=(1-lr)*q[arm]+lr*reward
 return reg
def run():
 matrix={s:{t:cosine(se,te) for t,te in TARGET_EFFECTS.items()} for s,se in SOURCE_EFFECTS.items()}
 max_cross=max(v for row in matrix.values() for v in row.values())
 discrimination={}
 for intent,e in INTENTS.items():
  vals=sorted((cosine(e,te),t) for t,te in TARGET_EFFECTS.items())
  discrimination[intent]={"best":vals[-1],"worst":vals[0],"delta_cos":vals[-1][0]-vals[0][0]}
 # phase niches: mechanism with lowest regret in each quarter, across seeds
 niches={}
 for phase,(a,b) in enumerate(((0,180),(180,360),(360,540),(540,720))):
  means={op:statistics.fmean(op_regret(s,op,a,b) for s in range(120)) for op in TARGET_EFFECTS}
  niches[str(phase)]={"means":means,"best":min(means,key=means.get)}
 distinct=len({x["best"] for x in niches.values()})
 # schema measurability contract: dimensions must be computable or maskable; current dataclass has all six numeric slots.
 out={"source_target_cosine":matrix,"max_cross_cosine":max_cross,"no_near_identity":max_cross<=.95,
      "intent_discrimination":discrimination,"min_delta_cos":min(x["delta_cos"] for x in discrimination.values()),
      "adapter_discriminative":min(x["delta_cos"] for x in discrimination.values())>=.10,
      "acquisition_state_dimensions":DIMS,"mask_required_for_cross_host":True,
      "phase_niches":niches,"distinct_best_mechanisms":distinct,"niches_nontrivial":distinct>=2}
 # Cross-host acquisition state must explicitly support masking unavailable dimensions.
 from acquisition_state import AcquisitionState
 probe=AcquisitionState(1,.5,.5,.1,.0,.0,(True,True,False,False,True,False))
 masked,mask=probe.masked()
 out["acquisition_mask_contract"]=len(mask)==len(DIMS) and len(masked)==len(DIMS) and any(not x for x in mask)
 out["prerequisites_pass"]=out["no_near_identity"] and out["adapter_discriminative"] and out["niches_nontrivial"] and out["acquisition_mask_contract"]
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/semantic_prerequisites.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
