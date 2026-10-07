#!/usr/bin/env python3
"""Minimal experiment harness: reproducible artifact dirs + invalid-experiment accounting."""
from __future__ import annotations
import json,subprocess,sys,time,hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ART=ROOT/"artifacts"
def run(name,command):
 ART.mkdir(parents=True,exist_ok=True);start=time.time()
 p=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
 rec={"schema":"symcore.experiment.v1","name":name,"command":command,"returncode":p.returncode,
      "status":"VALID_RUN" if p.returncode==0 else "INVALID_OR_FAILED","seconds":time.time()-start,
      "stdout_sha256":hashlib.sha256(p.stdout.encode()).hexdigest(),"stderr_tail":p.stderr[-2000:]}
 with (ART/"experiment_ledger.jsonl").open("a") as f:f.write(json.dumps(rec,sort_keys=True)+"\n")
 sys.stdout.write(p.stdout);sys.stderr.write(p.stderr);return p.returncode
if __name__=="__main__":
 if len(sys.argv)<3:raise SystemExit("usage: experiment_runner.py NAME COMMAND...")
 raise SystemExit(run(sys.argv[1],sys.argv[2:]))
