from VLEARN_PREREGISTRATION import *
def test_vlearn_zero_horizon_is_zero_by_definition():
 assert 0 not in HORIZONS and all(h>0 for h in HORIZONS)
def test_grid_is_frozen():
 assert FREEZE_POINTS==(1024,2048,4096,8192,16384) and HORIZONS==(256,512,1024,2048)
def test_analysis_is_not_mislabeled_confirmatory():
 assert ANALYSIS_STATUS.startswith("posthoc")
