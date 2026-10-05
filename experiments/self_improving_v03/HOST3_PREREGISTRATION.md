# Host 3 pre-registration
Paradigm: sequential decision/control, not forecasting.
Environment: non-stationary contextual bandit with hidden regime changes and observable causal state.

Pi_0 is the frozen current intervention policy. Pi_1 is learned from prior host-state / future-value histories. Neither may alter the external evidence gate.

Primary endpoint: on held-out unseen regimes, Pi_1 must reduce cumulative regret versus Pi_0 with a 95% confidence interval whose lower improvement bound is > 0, while satisfying the same risk budget.

This directly tests whether prior growth experience produced a better future growth policy: Pi_1 > Pi_0.
