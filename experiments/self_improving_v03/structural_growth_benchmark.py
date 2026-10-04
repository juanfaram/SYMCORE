#!/usr/bin/env python3
"""Real-time structural growth simulation: detect uncovered context -> create skill -> verify acquisition."""
import json,random
from pathlib import Path
from gap_detector import GapDetector
from capability_factory import CapabilityFactory
from skill_memory import SkillMemory
from capability_ledger import CapabilityLedger,LedgerEntry
def run(seed=1401):
 rng=random.Random(seed);actions=["fast","balanced","deep","creative"];det=GapDetector(.65,40);registry=[{"name":"general","context":{"domain":"known"}}]
 # Phase 1: observe a genuinely uncovered context
 ctx={"task":"solve","domain":"novel-X","difficulty":"hard"}
 for _ in range(80):det.observe(ctx,{"general":rng.uniform(.25,.5)})
 gaps=det.gaps();spec=CapabilityFactory().propose(gaps[0],registry)
 # Phase 2: instantiate bounded specialist and learn causally
 m=SkillMemory(actions);correct=[];target="deep"
 for i in range(600):
  a=m.choose(ctx);ok=a==target;m.feedback(a,ctx,1 if ok else -.3);correct.append(ok)
 before=sum(correct[:100])/100;after=sum(correct[-100:])/100
 survive=after>=.8 and after-before>=.3
 led=CapabilityLedger()
 led.append(LedgerEntry(1,"novel-X stream","no expert >= .65",spec.name,spec.parents,"novel-X-hard",
  {"before":before,"after":after,"gain":after-before},"SURVIVE" if survive else "REJECT",680,1.0,600))
 out={"gap":gaps[0],"proposal":spec.__dict__,"before":before,"after":after,"survived":survive,"ledger_valid":led.verify_chain()}
 Path("artifacts/structural_growth_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
