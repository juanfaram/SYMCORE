from symcore.policy import SymCorePolicy

def test_known_negative_regimes_bypass():
    p=SymCorePolicy()
    assert not p.should_attempt(1024,8.0)
    assert not p.should_attempt(1536,2.0)

def test_validated_positive_regimes_allowed():
    p=SymCorePolicy()
    assert p.should_attempt(1536,4.0)
    assert p.should_attempt(2048,8.0)
