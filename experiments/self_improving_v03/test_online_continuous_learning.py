from online_continuous_learning import OnlineMemory,Interaction,Checkpoint,evaluate_contract

def test_exactly_one_interaction_increments_experience_once():
 m=OnlineMemory();e=Interaction((1,), "a",1,.1,.8);m.update_one(e);assert m.n==1;m.update_one(e);assert m.n==2

def test_frozen_copy_does_not_change_when_live_updates():
 live=OnlineMemory();live.update_one(Interaction((1,),"a",1,.1,.8));frozen=live.frozen_copy();live.update_one(Interaction((1,),"a",2,.1,.7))
 assert live.n==2 and frozen.n==1 and frozen.value((1,),"a")==1

def test_online_memory_has_no_batch_training_surface():
 for forbidden in ("fit","retrain","replay","fit_batch","update_batch"):
  assert not hasattr(OnlineMemory,forbidden)

def test_longitudinal_contract_requires_positive_and_negative_slopes_and_live_advantage():
 cs=[Checkpoint(i,0.02*i,1-.03*i,.01*i) for i in range(1,7)]
 assert evaluate_contract(cs)["passed"]
