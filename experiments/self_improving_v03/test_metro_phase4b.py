from METRO_PHASE4B_PREREGISTRATION import *
from metro_real_host import MetroControl,MetroSymcore
def test_preregistration_is_chronological():
 assert FREEZE_AT<EVAL_START<CHECKPOINTS[0] and len(CHECKPOINTS)>=MIN_CHECKPOINTS and CHECKPOINTS[-1]<=N_EXPECTED
def test_three_arm_components_exist():
 assert hasattr(MetroControl(),"step") and hasattr(MetroSymcore(),"step")
def test_cost_is_vector_not_advantage_ratio():
 assert "seconds_per_interaction" in COST_AXES and "updates_per_interaction" in COST_AXES and "peak_bytes" in COST_AXES
