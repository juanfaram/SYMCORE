# SYMCORE — Experimental Capability Ecology v1.0

SYMCORE is an evidence-driven laboratory for continual capability acquisition.

Its core lifecycle is:

```
wild population
  -> replicated evidence
  -> adversarial / OOD examination
  -> uncertainty-aware validation
  -> capability frontier
  -> quality-diversity repertoire
  -> research-budget allocation
  -> new experiments
```

## What counts as progress

Not “more code” and not a single higher score. Progress means expanding reachable competent behavior while retaining validated capabilities at known cost.

## Current mechanisms

- causal online learning and drift detection
- evolutionary candidates and lineage
- contextual interaction learning
- seasonal specialists and adaptive mixtures
- replay / continual-learning curriculum
- adversarial and OOD arenas
- multi-seed replication
- protected-axis capability frontier
- novelty archive / MAP-Elites-like repertoire
- coevolutionary examiner
- evidence-driven research scheduler
- confidence-bound validation
- causal ablation tests
- hard no-regression CI gates

## Validation contract

Exploration is permissive; acceptance is deliberately difficult. A capability may be archived for novelty without entering the validated frontier. Important claims require replication across seeds and bounded uncertainty. CI preserves evidence even on failure.

## Run

```bash
cd experiments/self_improving_v03
python -m compileall -q .
pytest -q
python evaluate.py
python frontier_tournament.py
```

The system does not grant generated candidates permission to rewrite or deploy themselves. Experimental autonomy happens inside the evaluated capability lifecycle.
