#!/usr/bin/env python3
"""Standalone Capability Ledger verifier. Imports no SYMCORE runtime modules."""
import argparse,hashlib,json
from pathlib import Path
def verify(path):
 prev="GENESIS";n=0
 for line in Path(path).read_text().splitlines():
  if not line.strip():continue
  row=json.loads(line);claimed=row.pop("entry_hash");row["entry_hash"]=""
  actual=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()
  if row.get("prev_hash")!=prev or actual!=claimed:return False,n
  prev=claimed;n+=1
 return True,n
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("ledger");a=p.parse_args();ok,n=verify(a.ledger);print(json.dumps({"valid":ok,"entries":n}));raise SystemExit(0 if ok else 1)
