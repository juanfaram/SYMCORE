#!/usr/bin/env python3
"""Frozen Necessity Destruction contract for z_H geometry."""
FULL_DIMS=tuple(range(15))
MAX_VALUE_CAPTURE_DROP=0.002
MAX_MEAN_REGRET_RISE=0.02
MIN_HOSTS_IMPROVED_VS_EH=2
STATUS="development_on_opened_hosts"
RULE="delete only while removal is non-inferior within frozen tolerances; stop at first unjustified deletion"
