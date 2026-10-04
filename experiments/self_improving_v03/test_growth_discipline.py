import json
from capability_ledger import CapabilityLedger,LedgerEntry

def test_observation_does_not_count_as_verified_growth(tmp_path):
 l=CapabilityLedger(tmp_path/"ledger.jsonl")
 l.append(LedgerEntry(1,"x","future","candidate",[],"meta",{"ft":.01},"OBSERVE",10,1,5))
 assert l.verified()==[] and l.growth_rate()==0

def test_rejected_capability_cannot_inflate_growth(tmp_path):
 l=CapabilityLedger(tmp_path/"ledger.jsonl")
 l.append(LedgerEntry(1,"x","gap","bad",[],"bad-skill",{},"REJECT",10,1,2))
 l.append(LedgerEntry(2,"x","gap","good",[],"good-skill",{},"SURVIVE",10,1,2))
 assert len(l.verified())==1 and l.verify_chain()
