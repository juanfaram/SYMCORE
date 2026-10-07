from PI_LEARN_CONTEXT_CONTRACT import *
from learning_state import LearningState
def test_contract_is_paired_and_fail_closed():
 assert BASELINE=="Pi_learn(E,h)" and CANDIDATE=="Pi_learn(E,z_H,h)"
 assert CONTRACT["value_capture_improved_hosts_gte"]==2 and CONTRACT["mean_regret_reduced_hosts_gte"]==2
 assert CONTRACT["no_fourth_host_if_fail"] is True
def test_learning_state_uses_only_observed_loss_history():
 s=LearningState();before=s.vector();s.observe(10);s.observe(9);after=s.vector()
 assert len(before)==len(after)==3
