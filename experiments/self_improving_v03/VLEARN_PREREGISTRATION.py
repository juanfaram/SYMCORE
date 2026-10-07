#!/usr/bin/env python3
"""Preregister V_learn surface analysis on already-opened real hosts.
Exploratory/post-hoc with respect to hosts; frozen prospectively with respect to this analysis."""
FREEZE_POINTS=(1024,2048,4096,8192,16384)
HORIZONS=(256,512,1024,2048)
MIN_POSITIVE_FREEZE_POINTS=4
MIN_POSITIVE_HORIZONS=4
HOSTS=("bike","metro","beijing")
# Causal definition: identical state at E; only live continues updates during (E,E+h].
# Therefore V(E,0)=0 by construction.
CONTRACT={"positive_freeze_points_gte":4,"positive_horizons_per_freeze":4,"hosts_required":3}
ANALYSIS_STATUS="posthoc_on_opened_hosts_not_confirmatory_holdout"
