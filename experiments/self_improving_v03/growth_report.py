#!/usr/bin/env python3
"""Convert experiment reports into capability-growth evidence."""
import json
from pathlib import Path
from capability_ledger import CapabilityLedger,LedgerEntry
def run():
    mt=json.loads(Path("artifacts/meta_transfer_report.json").read_text())
    led=CapabilityLedger()
    ft=mt["forward_transfer"];survive=ft>0 and mt["experienced_mean"]<mt["scratch_mean"]
    led.append(LedgerEntry(len(led.entries)+1,"multi-seed meta-transfer","future acquisition cost","transfer-prior-v1",
      ["factorized-skill-memory"],"meta-transfer",{"forward_transfer":ft,"scratch_cost":mt["scratch_mean"],"experienced_cost":mt["experienced_mean"]},
      "SURVIVE" if survive else "REJECT",sum(mt["seeds"])*4001,1.0,mt["experienced_mean"]))
    out={"verified_capabilities":len({e["capability"] for e in led.verified()}),"G":led.growth_rate(),"L":led.learning_acceleration(),"chain_valid":led.verify_chain()}
    Path("artifacts/growth_report.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
