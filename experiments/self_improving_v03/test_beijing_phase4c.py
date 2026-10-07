from BEIJING_PHASE4C_PREREGISTRATION import *
from beijing_real_host import BeijingControl,BeijingSymcore
def test_f4c_is_frozen_chronologically():
 assert FREEZE_AT<EVAL_START<CHECKPOINTS[0] and len(CHECKPOINTS)>=5 and CHECKPOINTS[-1]<N_EXPECTED
def test_weekly_bootstrap_is_preregistered():
 assert BLOCK==168 and BOOTSTRAPS>=1000
def test_three_arm_online_components_exist():
 assert hasattr(BeijingControl(),"step") and hasattr(BeijingSymcore(),"step")
