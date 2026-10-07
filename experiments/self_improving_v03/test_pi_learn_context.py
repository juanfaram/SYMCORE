from learning_state import LearningState
from pi_learn_value import score
def test_learning_state_is_three_dimensional_and_finite():
 s=LearningState()
 for i in range(300):s.observe(10/(1+i/20))
 z=s.vector();assert len(z)==3 and all(__import__("math").isfinite(x) for x in z)
def test_value_metric_penalizes_missing_large_positive_more():
 a=score([("FREEZE",10.),("LEARN",1.)]);b=score([("LEARN",10.),("FREEZE",1.)])
 assert b["value_capture"]>a["value_capture"] and b["mean_regret"]<a["mean_regret"]
