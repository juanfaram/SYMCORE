# SYMCORE Host Interface

SYMCORE is a universal continuous-growth engine. Its host contract is deliberately smaller than any particular model or task.

## Host contract

A host exposes four operations:

- `predict(x)`: produce an output without observing the current target.
- `feedback(x, y)`: update only after the outcome is revealed.
- `loss(prediction, y)`: map host-specific performance to a non-negative comparable cost.
- `capabilities()`: expose the currently available behavioral capabilities and evidence.

An adaptive host may additionally expose `candidates()`, a mapping of bounded alternative host behaviors that SYMCORE may evaluate or route between.

SYMCORE must not depend on the host's model class, parameters, dataset schema, or domain. Host-specific adapters translate those details into this contract.

## Causal control

Every product-level evaluation pairs two initially equivalent hosts on the same ordered experience stream:

`control` receives normal host learning; `assisted` receives the same observations plus SYMCORE interventions.

Prediction always occurs before feedback. Define host advantage as:

`A = log(cost_control / cost_assisted)`.

A > 0 means SYMCORE helped; A = 0 means no measured contribution; A < 0 means it harmed the host.

The control is part of the operating contract, not merely a benchmark convenience.

## Growth contract

The engine may claim strong progress only when evidence supports all four conditions:

1. ΔΩ > 0 — reachable verified capability expands.
2. A > 0 — the assisted host outperforms its paired control.
3. dA/dE > 0 — advantage increases with accumulated experience.
4. P(R > Rmax) < δ — regressions remain bounded.

A host adapter is valid only if the engine core can use it without host-specific branches in the engine.
