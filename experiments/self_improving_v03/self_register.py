#!/usr/bin/env python3
"""Level-1 self-application: register SYMCORE development candidates without autonomous promotion."""
import json,os,time,hashlib
from pathlib import Path
def register(candidate,sha,evidence=None):
 p=Path("artifacts/self_development_ledger.jsonl");p.parent.mkdir(exist_ok=True)
 prev="GENESIS"
 if p.exists():
  rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
  if rows:prev=rows[-1]["entry_hash"]
 row={"timestamp":time.time(),"candidate":candidate,"commit_sha":sha,"evidence":evidence or {},
      "decision":"OBSERVE_ONLY","autonomous_promotion":False,"prev_hash":prev}
 row["entry_hash"]=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
 with p.open("a") as f:f.write(json.dumps(row,sort_keys=True)+"\n")
 return row
if __name__=="__main__":
 print(json.dumps(register(os.getenv("GITHUB_REF_NAME","unknown"),os.getenv("GITHUB_SHA","unknown")),indent=2))
