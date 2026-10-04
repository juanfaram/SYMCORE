from representation import StructuralRepresentation
from representation_learner import RepresentationLearner
def test_representation_learns_factor_relevance():
 r=StructuralRepresentation()
 for _ in range(20):r.observe({"domain":"x","difficulty":"hard","task":"t"},1)
 assert abs(sum(r.relevance({"domain":"x","difficulty":"hard","task":"t"}).values())-1)<1e-9
def test_representation_learner_runs():
 r=StructuralRepresentation();a=RepresentationLearner(["x","y"],r);c={"domain":"d","difficulty":"q","task":"t"}
 a.feedback("y",c,1);assert a.choose(c) in ("x","y")
