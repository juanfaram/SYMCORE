# SYMCORE — Persistent Capability Growth Engine (experimental)

North star:

```
experience -> better capabilities -> better learner -> cheaper future acquisition -> more capabilities
```

We model effective capability as a repertoire, not a single score. The system is designed to progress through:

1. adaptation
2. acquisition
3. conservation
4. composition
5. meta-learning
6. evolution of acquisition strategies

## Core real-use loop

```
interaction
 -> context / decision / outcome
 -> hierarchical memory
 -> weakness / novelty / drift detection
 -> online adaptation
 -> capability-gap detection
 -> bounded specialist generation / mutation / recombination
 -> adversarial + OOD + multi-seed examination
 -> evidence archive
 -> validated frontier / repertoire
 -> Capability Ledger
 -> future-learning priors
 -> next interaction
```

Fast operations (memory, confidence, routing) can update per interaction. Structural growth requires accumulated evidence.

## Capability Ledger

The ledger records generation, experience source, weakness, candidate, parents, new capability, evidence, decision, acquisition cost and lineage hash. This makes it possible to reconstruct how the system learned to learn.

## Growth metrics

`G(t)`: verified new capabilities per experience × compute.

`L(t)`: decrease in acquisition cost as experience accumulates.

A system that merely improves accuracy but does not expand verified reachable competence is not considered structurally growing.

## Current evidence

The multi-seed meta-transfer experiment has demonstrated a large reduction in acquisition interactions versus scratch. This evidence is now ingested into the ledger rather than treated as an isolated benchmark.

Exploration remains broad; promotion remains evidence-gated.
