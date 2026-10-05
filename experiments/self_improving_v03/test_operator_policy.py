from operator_policy import OperatorPolicy,OPERATORS
def test_pi_starts_uniform_and_learns_without_collapsing():
 p=OperatorPolicy();d0=p.distribution("drift")
 assert len(set(round(x,8) for x in d0.values()))==1
 for _ in range(5):p.update("drift","memory",True)
 for _ in range(3):p.update("drift","parameters",False)
 d1=p.distribution("drift")
 assert d1["memory"]>d1["parameters"]
 assert all(d1[o]>0 for o in OPERATORS)
