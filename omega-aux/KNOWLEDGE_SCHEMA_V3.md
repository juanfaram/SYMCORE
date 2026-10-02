# Knowledge Schema v3

Evidence becomes operational KNOWLEDGE only when it changes a future policy and the future outcome is later observed.
Required additions: policy_change, future_outcome_verified.

If policy_change is absent: DATA.
If policy_change exists and future outcome worsens: NEGATIVE_TRANSFER.
If policy_change exists and future outcome improves: LEARNING.
If future outcome not yet observed: KNOWLEDGE_CANDIDATE.

PATTERN still requires >=2 independent empirical cases.
