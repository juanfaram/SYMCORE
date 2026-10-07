#!/usr/bin/env python3
"""Frozen preregistration: third independent real host, Beijing PM2.5."""
DATASET="UCI Beijing PM2.5";UCI_ID=381;N_EXPECTED=43824
FREEZE_AT=8760
EVAL_START=17520
CHECKPOINTS=(20400,23300,26200,29100,32000,34900,37800,40700,43600)
BLOCK=168 # one week of hourly observations
BOOTSTRAPS=3000
CONTRACT={"dA_dE_lcb95_gt":0.0,"live_gt_frozen_rate_gte":0.80}
COST_AXES=("seconds_per_interaction","updates_per_interaction","peak_bytes")
