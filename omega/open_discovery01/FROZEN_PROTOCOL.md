# OPEN-DISCOVERY-01 — frozen protocol

Target: construct a 17-channel oblivious sorting network with <=70 comparators.

Frozen external incumbent:
- 71 comparators, 12 layers, Baddar09 / Dobbelaere compilation.
- External specialist: SorterHunter.
- Primary public source: https://bertdobbelaere.github.io/sorting_networks.html

Success:
- one submitted network with <=70 comparators;
- exhaustive zero-one verification over all 2^17 inputs;
- independent re-verification before any novelty claim.

Failure:
- NO DISCOVERY WITHIN SEARCH SCOPE is valid.

Search budget v1:
- 4 independent shards;
- 45 minutes maximum per shard;
- fixed random seeds 17001..17004;
- no changing fitness/search operators after run begins.

Algorithm space:
- complete comparator networks;
- comparator deletion;
- comparator replacement;
- endpoint mutation;
- comparator relocation;
- local segment replacement;
- counterexample-guided mutations;
- recombination may be added only in a later frozen version.

Fitness:
1. number of unsorted binary inputs (exact, all 131072);
2. comparator count;
3. deterministic tie-break.

Blind:
The exhaustive verifier is the mathematical oracle, so there is no statistical blind set.
Anti-Goodhart requirement: a candidate must be one fixed oblivious network and must sort every 17-bit input.

Novelty:
CANDIDATE only if <=70 and exhaustive verification passes.
FULL only after external duplicate/literature check and independent verification/publication.
