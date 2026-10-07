#!/usr/bin/env python3
"""Independent real-host replication preregistration: UCI Metro Interstate Traffic Volume."""
DATASET="UCI Metro Interstate Traffic Volume"
UCI_ID=492
N_EXPECTED=48204
# Fixed fractions chosen before outcomes: freeze ~20%, evaluation begins ~40%.
FREEZE_AT=9600
EVAL_START=19200
CHECKPOINTS=(22400,25600,28800,32000,35200,38400,41600,44800,48000)
MIN_CHECKPOINTS=5
CONTRACT={"dA_dE_lcb95_gt":0.0,"live_gt_frozen_rate_gte":0.80}
# Cost is now a vector; no arbitrary scalar gate until external prices/weights are justified.
COST_AXES=("seconds_per_interaction","updates_per_interaction","peak_bytes")
