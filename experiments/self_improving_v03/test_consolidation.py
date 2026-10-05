from evidence_status import EvidenceStatus
def test_verified_requires_all_three():
 assert EvidenceStatus(True,True,True).decision()=="VERIFIED_CAPABILITY"
 assert EvidenceStatus(True,True,False).decision()=="UNVERIFIED"
