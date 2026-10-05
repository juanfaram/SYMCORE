from risk_contract import risk_report,passes
from acceleration_contract import acceleration,replicated
from invariant_contract import NorthStar
def test_risk_contract_distinguishes_safe_growth():
 r=risk_report([10]*500,[9]*500,.05);assert passes(r,.05)
 r=risk_report([10]*500,[12]*500,.05);assert not passes(r,.05)
def test_acceleration_needs_three_points():
 assert not acceleration([100,50])["passed_dA"]
 assert acceleration([100,80,50])["passed_A"]
def test_unknown_is_not_success():
 s=NorthStar(delta_omega=True,A=None,dA_dE=None,risk=True).status();assert not s["all_proven"] and "A" in s["unknown"]
