from intervention_controller import InterventionController
def test_controller_rejects_outlier_and_accepts_sustained_shift():
 c=InterventionController(enter_z=.5,exit_z=.1,min_history=32,window=64,exit_patience=3)
 # Nonzero baseline variance makes the z-score test meaningful.
 for i in range(48):c.observe(1.0 + (0.1 if i%2 else -0.1))
 assert c.decide() is False
 # One spike is explicitly not enough.
 c.observe(4.0)
 assert c.decide() is False
 # A sustained recent regime shift is enough, using only already-observed losses.
 for _ in range(23):c.observe(4.0)
 assert c.decide() is True
 # Hysteresis: one calm observation cannot switch it straight back off.
 c.observe(1.0)
 assert c.decide() is True
