#!/usr/bin/env python3
"""Causal policy-value scoring for LEARN/FREEZE/MEASURE."""
from __future__ import annotations
def action_value(action,v_learn,measure_cost=0.25):
 if action=="LEARN":return float(v_learn)
 if action=="FREEZE":return 0.
 if action=="MEASURE":return max(float(v_learn),0.)-float(measure_cost)
 raise ValueError(action)
def oracle_action(v_learn,measure_cost=0.25):
 # With realized V known, measuring has no oracle advantage; it exists for uncertain policy decisions.
 return "LEARN" if v_learn>0 else "FREEZE"
def score(decisions,measure_cost=0.25):
 # decisions: iterable[(chosen_action, realized_v)]
 rows=list(decisions);vals=[action_value(a,v,measure_cost) for a,v in rows];opt=[max(float(v),0.) for _,v in rows]
 captured=sum(vals);oracle=sum(opt);regret=sum(o-v for o,v in zip(opt,vals))
 return {"policy_value":captured,"oracle_value":oracle,"value_capture":captured/oracle if oracle>0 else 1.,
         "mean_regret":regret/max(1,len(rows)),"positive_value_rate":sum(x>0 for x in vals)/max(1,len(vals))}
