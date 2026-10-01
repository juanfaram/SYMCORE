# Ω Deep Optimization Policy

## Mission

Optimize for maximum verified software improvement, not minimum optimizer wall-clock time.

A run may consume hours or days if expected improvement justifies it. Search cost is measured and reported, but it is not the primary objective unless the contract says so.

## Search hierarchy

Every problem is attacked from highest leverage to lowest:

1. Contract / observability reduction
2. Semantic representation lift
3. Algorithmic complexity change
4. Data-structure / layout change
5. Fusion, batching, caching, memoization
6. Parallel/vector/hardware mapping
7. Compiler/IR transformations
8. Algebraic/local rewrites
9. Micro-optimizations

A lower layer must not monopolize budget while a higher layer has unresolved high-value hypotheses.

## Three necessity levels

- G_N_local: what this implementation needs.
- G_N_algorithm: what this algorithmic family needs.
- G_N_contract: what any admissible realization needs.

Primary optimization opportunity:
- Δ_impl = G_N_local - G_N_algorithm
- Δ_alg = G_N_algorithm - G_N_contract
- Δ_proof = suspected removable work - justified removable work

## Candidate lifecycle

HYPOTHESIS -> ATTACKED -> QUARANTINED/REJECTED/VERIFIED -> MEASURED -> PARETO

No candidate skips attack.
No measured speedup upgrades semantic evidence.
No proof of equivalence upgrades performance evidence.

## Deep-search stopping rule

Stop only when one of these holds:
- a valid lower bound is reached to the declared resolution;
- all higher-leverage representation/algorithmic hypotheses are exhausted under the declared grammar;
- expected value of remaining search falls below its resource budget;
- contractual budget is exhausted.

"Compiler produced no further improvement" is never a stopping proof.

## Portfolio (future activation after Iteration 3)

Representation recognizers:
- reduction/fold
- scan/prefix
- map/filter/fusion
- search/membership
- recurrence/dynamic programming
- dense linear algebra
- sparse/graph
- streaming/state machine

Search families:
- rewrite
- algebraic
- algorithmic
- structural/data representation
- parametric
- enumerative
- stochastic
- compiler/oracle-backed

## Evidence ledger

Every claim carries one or more:
[P] proof
[M] measured locally
[T] tested
[I] inferred
[E] external oracle
[H] hypothesis

The ledger is dimension-specific: correctness, contract, performance, memory, effects, exceptions, termination and portability are separate.

## Client deliverable

A client optimization is complete only with:
- optimized source/artifact
- patch/diff
- contract and assumptions
- correctness evidence
- multiscale performance evidence
- lower bound / class gap when available
- remaining UNKNOWNs
- reproduction command
- rollback path
