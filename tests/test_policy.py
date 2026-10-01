from symcore.policy import SymCorePolicy

def test_l1024_r8_bypass():
    assert not SymCorePolicy().should_attempt(1024, 8.0)

def test_l1536_r2_bypass():
    assert not SymCorePolicy().should_attempt(1536, 2.0)

def test_l1536_r4_attempt():
    assert SymCorePolicy().should_attempt(1536, 4.0)

def test_l2048_r8_attempt():
    assert SymCorePolicy().should_attempt(2048, 8.0)
