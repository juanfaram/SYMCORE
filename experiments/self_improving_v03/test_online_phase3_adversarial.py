from online_phase3_adversarial import one,CHECKPOINTS,LEVELS,REQUIRED
def test_all_conditions_share_identical_checkpoint_budget():
 for k,levels in LEVELS.items():
  for v in levels:assert tuple(x[0] for x in one(0,k,v))==CHECKPOINTS
def test_zero_adversaries_do_not_change_stream_contract_shape():
 xs=[one(7,k,0) for k in LEVELS]
 assert all(len(x)==len(CHECKPOINTS) for x in xs)
def test_unlock_thresholds_were_preregistered_inside_tested_grid():
 for k,v in REQUIRED.items():assert v in LEVELS[k]
