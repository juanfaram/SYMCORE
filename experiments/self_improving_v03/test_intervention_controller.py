from intervention_controller import InterventionController
def test_controller_is_causal_and_hysteretic():
 c=InterventionController(enter_z=.5,exit_z=.1,min_history=8,window=16,exit_patience=3)
 for x in [1.0]*8:c.observe(x)
 assert c.decide() is False
 # A current outcome cannot affect the decision until after observe().
 before=c.decide()
 c.observe(10.0)
 assert before is False
 # The design intentionally reacts to sustained recent change, not one spike.
 for _ in range(7):c.observe(10.0)
 assert c.decide() is True
 # One calm observation cannot chatter the controller back off.
 c.observe(1.0)
 assert c.decide() is True
