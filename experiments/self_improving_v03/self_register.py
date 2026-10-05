#!/usr/bin/env python3
"""Write-only Level-1 self-development registration. No read/query API exists in this module."""
import json,os,time,hashlib
from pathlib import Path
def register(candidate,sha,evidence=None,prev_hash="EXTERNAL"):
 p=Path("artifacts/self_development_ledger.jsonl");p.parent.mkdir(exist_ok=True)
 row={"timestamp":time.time(),"candidate":candidate,"commit_sha":sha,"evidence":evidence or {},"decision":"OBSERVE_ONLY",
      "autonomous_promotion":False,"prev_hash":prev_hash}
 row["entry_hash"]=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
 with p.open("a") as f:f.write(json.dumps(row,sort_keys=True)+"\n")
 return {"written":True,"entry_hash":row["entry_hash"],"decision":"OBSERVE_ONLY"}
if __name__=="__main__":print(json.dumps(register(os.getenv("GITHUB_REF_NAME","unknown"),os.getenv("GITHUB_SHA","unknown")),indent=2))
