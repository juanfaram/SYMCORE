#!/usr/bin/env python3
"""External-only self-development ledger metrics. Not imported by runtime self-registration."""
import json,statistics
from pathlib import Path
def run():
 p=Path("artifacts/self_development_ledger.jsonl");rows=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
 verified=[r for r in rows if r.get("evidence",{}).get("verified") is True]
 costs=[r.get("evidence",{}).get("cost") for r in rows if isinstance(r.get("evidence",{}).get("cost"),(int,float))]
 out={"proposals":len(rows),"verified":len(verified),"G_development":len(verified)/max(1,len(rows)),
      "mean_acquisition_cost":statistics.fmean(costs) if costs else None,"observe_only":all(r.get("decision")=="OBSERVE_ONLY" for r in rows)}
 Path("artifacts").mkdir(exist_ok=True);Path("artifacts/self_development_metrics.json").write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return out
if __name__=="__main__":run()
