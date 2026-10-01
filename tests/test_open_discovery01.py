from omega.open_discovery01.incumbent import INCUMBENT_71
from omega.open_discovery01.verifier import evaluate,verify

def test_incumbent_is_valid():
    assert len(INCUMBENT_71)==71
    assert evaluate(INCUMBENT_71)==0

def test_every_single_deletion_fails():
    # Confirms the incumbent has no trivially redundant comparator.
    assert all(evaluate(INCUMBENT_71[:i]+INCUMBENT_71[i+1:])>0 for i in range(71))
