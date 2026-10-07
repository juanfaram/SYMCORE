from transfer_prior import TransferPrior
from skill_memory import SkillMemory
from seasonal import SeasonalMemory
from residual_specialist import ResidualSpecialist
def test_transfer_prior_seeds_related_context():
 p=TransferPrior(["a","b"]);c={"domain":"x","difficulty":"hard"}
 for _ in range(20):p.observe(c,"b",1)
 m=SkillMemory(["a","b"]);p.seed(m,c,8);assert m.choose(c)=="b"
def test_residual_specialist_runs():
 a=SeasonalMemory(("hr",),4);m=ResidualSpecialist(a);r={"hr":"1","workingday":"1"}
 m.learn(r,10);assert m.predict(r)>=0
