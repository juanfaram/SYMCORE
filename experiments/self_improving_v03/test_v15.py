import json
from standalone_ledger_verify import verify
from blind_gate import BlindEvidence,evaluate
from lifecycle_policy import Lifecycle
from capability_ledger import CapabilityLedger,LedgerEntry
def test_independent_ledger_verifier(tmp_path):
 p=tmp_path/"l.jsonl";l=CapabilityLedger(p);l.append(LedgerEntry(1,"x","w","c",[],"k",{},"SURVIVE",1,1,1))
 assert verify(p)==(True,1)
 rows=p.read_text().splitlines();x=json.loads(rows[0]);x["decision"]="REJECT";p.write_text(json.dumps(x)+"\n");assert verify(p)[0] is False
def test_blind_gate_has_no_origin_field():
 assert evaluate(BlindEvidence(.8,.8,.8,.8,.8,7))
 assert "origin" not in BlindEvidence.__dataclass_fields__
def test_hibernation_and_archive():
 l=Lifecycle(10,20);l.touch("a",0);assert l.update(11)["a"]=="hibernated";assert l.update(21)["a"]=="archived"
