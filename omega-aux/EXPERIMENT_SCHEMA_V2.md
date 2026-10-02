# Experiment Schema v2

NO RUN unless all five are answered before execution:
Q1 What do we actually know?
Q2 What is the largest uncertainty that matters?
Q3 What is the cheapest experiment that discriminates it?
Q4 Which result changes which decision?
Q5 What evidence is sufficient to stop?

Canonical outputs: CERTIFIED_IMPROVEMENT | REFUTED_CANDIDATE | KNOWLEDGE_GAIN | STOP_CERTIFICATE | INCONCLUSIVE.

Required identity: experiment_id, repo, branch, SHA, workflow, runID when assigned, H/T/A/E references, frozen thresholds, state=(Execution,Semantics,Effect,Verification).
