from capability_ledger import CapabilityLedger,LedgerEntry
from gap_detector import GapDetector
from capability_factory import CapabilityFactory
def test_ledger_chain_and_growth(tmp_path):
 l=CapabilityLedger(tmp_path/"l.jsonl")
 for i,cost in enumerate((100,60),1):l.append(LedgerEntry(i,"x","w",f"c{i}",[],f"k{i}",{},"SURVIVE",10,2,cost))
 assert l.verify_chain() and l.growth_rate()>0 and l.learning_acceleration()>0
def test_gap_creates_specialist():
 g=GapDetector(.65,2)
 c={"task":"x","domain":"new","difficulty":"hard"}
 for _ in range(3):g.observe(c,{"a":.2,"b":.4})
 gap=g.gaps()[0];s=CapabilityFactory().propose(gap,[{"name":"old","context":{"difficulty":"hard"}}])
 assert s.context["domain"]=="new" and "old" in s.parents
