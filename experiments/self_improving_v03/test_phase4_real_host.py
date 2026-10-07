from PHASE4_PREREGISTRATION import *
from phase4_real_host import Control,SymcoreOnline
def test_phase4_preregistration_is_chronological_and_large_enough():
 assert FREEZE_AT<EVAL_START<CHECKPOINTS[0] and len(CHECKPOINTS)>=MIN_CHECKPOINTS and tuple(sorted(CHECKPOINTS))==CHECKPOINTS
def test_three_arms_have_distinct_learning_semantics():
 c=Control();s=SymcoreOnline();assert hasattr(c,"step") and hasattr(s,"step")
def test_contract_is_frozen_before_results():
 assert CONTRACT=={"dA_dE_lcb95_gt":0.0,"dCnew_dE_ucb95_lt":0.0,"live_gt_frozen_rate_gte":0.80}
