from necessity_destroy_geometry import necessity_destroy
from GEOMETRY_PRUNING_CONTRACT import *
def test_pruning_stops_when_next_deletion_hurts():
 def ev(k):
  n=len(k);return {"a":{"value_capture":1-(3-n)*.01,"mean_regret":(3-n)*.1}}
 r=necessity_destroy((0,1,2),ev,tol_value=.002,tol_regret=.02)
 assert r["survivors"]==(0,1,2) and r["stop"]
def test_contract_starts_from_full_520_geometry():
 assert FULL_DIMS==tuple(range(15)) and MAX_VALUE_CAPTURE_DROP==.002 and MAX_MEAN_REGRET_RISE==.02
