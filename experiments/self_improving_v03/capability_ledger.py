#!/usr/bin/env python3
"""SYMCORE v1.4 Capability Ledger: append-only evidence of verified growth."""
from __future__ import annotations
import json,time,hashlib
from dataclasses import dataclass,asdict
from pathlib import Path
@dataclass
class LedgerEntry:
    generation:int;experience:str;weakness:str;candidate:str;parents:list;capability:str
    evidence:dict;decision:str;experience_units:float;compute_units:float;acquisition_cost:float
    timestamp:float=0.;prev_hash:str="";entry_hash:str=""
class CapabilityLedger:
    def __init__(self,path="artifacts/capability_ledger.jsonl"):
        self.path=Path(path);self.entries=[]
        if self.path.exists():
            self.entries=[json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]
    def append(self,e:LedgerEntry):
        e.timestamp=e.timestamp or time.time();e.prev_hash=self.entries[-1]["entry_hash"] if self.entries else "GENESIS"
        payload=asdict(e);payload["entry_hash"]=""
        e.entry_hash=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        row=asdict(e);self.entries.append(row);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open("a") as f:f.write(json.dumps(row,sort_keys=True)+"\n")
        return row
    def verified(self):return [e for e in self.entries if e["decision"]=="SURVIVE"]
    def growth_rate(self):
        xs=self.verified();caps=len({e["capability"] for e in xs});exp=sum(e["experience_units"] for e in xs);comp=sum(e["compute_units"] for e in xs)
        return caps/max(1e-9,exp*comp)
    def learning_acceleration(self):
        xs=self.verified()
        if len(xs)<2:return 0.
        costs=[e["acquisition_cost"] for e in xs]
        # positive when acquisition cost trends down with accumulated experience
        return (costs[0]-costs[-1])/max(1e-9,sum(e["experience_units"] for e in xs))
    def verify_chain(self):
        prev="GENESIS"
        for row in self.entries:
            x=dict(row);h=x.pop("entry_hash");x["entry_hash"]=""; 
            if x["prev_hash"]!=prev or hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()!=h:return False
            prev=h
        return True
