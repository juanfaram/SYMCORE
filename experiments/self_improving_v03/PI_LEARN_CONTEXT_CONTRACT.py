#!/usr/bin/env python3
"""Frozen development contract: does prospective z_H add transferable causal value?"""
HOSTS=("bike","metro","beijing")
CONTRACT={
 "value_capture_improved_hosts_gte":2,
 "mean_regret_reduced_hosts_gte":2,
 "no_fourth_host_if_fail":True
}
BASELINE="Pi_learn(E,h)"
CANDIDATE="Pi_learn(E,z_H,h)"
STATUS="development_leave_one_host_out_not_new_holdout"
