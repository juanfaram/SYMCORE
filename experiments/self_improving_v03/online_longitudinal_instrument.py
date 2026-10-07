#!/usr/bin/env python3
"""Instrument-only longitudinal harness: validates online causality before attaching a real host."""
import json,random
from pathlib import Path
from online_continuous_learning import OnlineMemory,Interaction,Checkpoint,evaluate_contract
CHECKPOINTS=(100,200,300,400,500,600)
def run():
 rng=random.Random(731001);live=OnlineMemory();control=OnlineMemory();frozen=None;rows=[]
 # Instrument stream only. Explicitly NOT evidence of real-host learning.
 for t in range(1,601):
  phase=t/600;z=(t%7,);action="adapt" if t%3 else "consolidate"
  # Learning opportunity becomes reusable: reward advantage rises; acquisition cost falls with accumulated online evidence.
  base=.45+rng.gauss(0,.03);gain=.00045*live.n
  live_reward=base+gain;control_reward=base
  cnew=max(.15,1.0-.0009*live.n)
  e=Interaction(z,action,live_reward,.1,cnew);live.update_one(e)
  control.update_one(Interaction(z,action,control_reward,.1,1.0))
  if t==100:frozen=live.frozen_copy()
  if t in CHECKPOINTS:
   frozen_perf=(base+.00045*frozen.n) if frozen else live_reward
   rows.append(Checkpoint(t,live_reward-control_reward,cnew,live_reward-frozen_perf))
 out=evaluate_contract(rows)
 out.update({"schema":"symcore.online-instrument.v1","instrument_only":True,"real_host_evidence":False,"checkpoints":[c.__dict__ for c in rows]})
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/online_instrument_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
