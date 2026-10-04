#!/usr/bin/env python3
"""Consolidate heterogeneous evidence into an auditable capability-growth ledger."""
import json
from pathlib import Path
from capability_ledger import CapabilityLedger,LedgerEntry
A=Path("artifacts")
def load(n):
 p=A/n
 return json.loads(p.read_text()) if p.exists() else None
def add(led,experience,weakness,candidate,parents,capability,evidence,decision,experience_units,compute,acquisition):
 return led.append(LedgerEntry(len(led.entries)+1,experience,weakness,candidate,parents,capability,
  evidence,decision,experience_units,compute,acquisition))
def run():
 led=CapabilityLedger()
 mt=load("meta_transfer_report.json");fr=load("frontier_report.json")
 rr=load("resurrection_report.json");res=load("residual_report.json");base=load("baseline_report.json")
 # Multi-seed frontier survivors are already validated by their dedicated evaluator.
 if fr:
  for name in fr.get("accepted_frontier",[]):
   e=fr["candidates"][name]
   add(led,"multi-seed frontier","expand capability manifold",name,[],"frontier:"+name,
       {"axes":e["axes"],"seeds":fr["seeds"]},"SURVIVE",len(fr["seeds"])*6000,1.0,6000)
 # Composition earns survival only if it solves the OOD composition test.
 if rr:
  acc=rr.get("compositional_ood_accuracy",0)
  add(led,"resurrection+OOD","compose prior skills","skill-composer",
      [x["skill"] for x in rr.get("resurrection",[])],"compositional-ood",{"accuracy":acc},
      "SURVIVE" if acc>=.60 else "REJECT",1000,1.0,1000)
 # Prediction specialists must beat a causal reference; no free promotion for novelty.
 if res and base:
  ref=min(base["persistence_mae"],base["same_hour_mae"]);mae=res["residual_mae_last512"]
  add(led,"bike stream","prediction residual","residual-specialist",[],"residual-prediction",
      {"mae":mae,"reference_mae":ref},"SURVIVE" if mae<ref else "REJECT",17379,1.0,17379)
 # Meta-transfer stays observational until repeated evidence supports a calibrated threshold.
 if mt:
  ft=mt["forward_transfer"];decision="OBSERVE"
  add(led,"multi-seed meta-transfer","future acquisition cost","transfer-prior-v1",
      ["factorized-skill-memory"],"meta-transfer",
      {"forward_transfer":ft,"scratch_cost":mt["scratch_mean"],"experienced_cost":mt["experienced_mean"],
       "seeds":mt["seeds"]},decision,len(mt["seeds"])*4001,1.0,mt["experienced_mean"])
 out={"schema":"symcore.growth.v2","verified_capabilities":len({e["capability"] for e in led.verified()}),
      "G":led.growth_rate(),"L":led.learning_acceleration(),"chain_valid":led.verify_chain(),
      "decisions":{d:sum(e["decision"]==d for e in led.entries) for d in ("SURVIVE","OBSERVE","REJECT")}}
 (A/"growth_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
