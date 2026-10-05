from intervention_controller import InterventionController
def test_controller_is_causal_and_hysteretic():
 c=InterventionController(enter_z=.5,exit_z=.1,min_history=8,window=16,exit_patience=3)
 for x in [1.0]*8:c.observe(x)
 assert c.decide() is False
 # Current loss cannot affect a decision until it is observed.
 before=c.decide();c.observe(10.0);after=c.decide()
 assert before is False and after is True
 # One calm observation is insufficient to chatter back off.
 c.observe(1.0)
 assert c.decide() is True
