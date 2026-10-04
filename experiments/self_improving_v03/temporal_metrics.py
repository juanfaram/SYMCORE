#!/usr/bin/env python3
"""Temporal metrics: regret, adaptation latency, recovery and forward transfer."""
def adaptation_latency(correct,window=100,target=.8):
    for i in range(window,len(correct)+1):
        if sum(correct[i-window:i])/window>=target:return i
    return len(correct)+1
def cumulative_regret(rewards,oracle=1.0):return sum(oracle-r for r in rewards)
def retention_loss(before,after):return max(0.,before-after)
def forward_transfer(baseline_latency,experienced_latency):
    return (baseline_latency-experienced_latency)/max(1,baseline_latency)
