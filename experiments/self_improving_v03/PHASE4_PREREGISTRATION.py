#!/usr/bin/env python3
"""Phase 4 preregistration: real external Bike Sharing stream, live/frozen/control."""
# Frozen before any Phase-4 outcome is inspected.
DATASET="UCI Bike Sharing hour.csv"
N_EXPECTED=17379
FREEZE_AT=4096
# Final chronological segment is untouched until the run. Six checkpoints inside final evaluation region.
EVAL_START=8192
CHECKPOINTS=(9216,10240,11264,12288,13312,14336,15360,16384,17379)
MIN_CHECKPOINTS=5
BLOCK=512
CONTRACT={
 "dA_dE_lcb95_gt":0.0,
 "dCnew_dE_ucb95_lt":0.0,
 "live_gt_frozen_rate_gte":0.80,
}
