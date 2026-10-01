# OPEN DISCOVERY-01 v1 — IMPROVEMENT CERTIFICATE

Canonical experiment:
- repo: juanfaram/SYMCORE
- branch: omega/open-discovery-01-sort17
- run: 36877418452
- SHA: 6bb1c1738c9f34d28b842ee647da5a97a108541b
- workflow conclusion: success

Objective:
17-channel sorting network with <=70 comparators.
Frozen incumbent: 71 comparators.
Exact fitness/verifier: number of failing 17-bit zero-one inputs.

Final shard results:
- seed 17001: bad=2; evaluated=6,110,552; accepted=1,725,271
- seed 17002: bad=2; evaluated=11,785,254; accepted=3,328,956
- seed 17003: bad=2; evaluated=8,742,406; accepted=2,485,374
- seed 17004: bad=2; evaluated=5,616,434; accepted=1,596,306

Total candidate evaluations: 32,254,646.
Verified <=70 sorting network found: NO.
Functional improvement over 71-comparator incumbent: NO.

Judgment:
NO MEJORA EN ESTE SCOPE — v1.

Important knowledge:
- all four independent shards reached exact distance bad=2;
- best distance was reached early (3,703 / 27,614 / 16,482 / 25,436 evaluations respectively) and no shard escaped it over millions more evaluations;
- this is strong evidence of a plateau/basin for the frozen local mutation + annealing strategy, not proof that a 70-comparator network does not exist.

Novelty: NONE.
Evidence: E3 for the behavior of this frozen search configuration; no claim about the global existence/nonexistence of a 70-comparator network.

Next allowed action:
analyze the bad=2 counterexamples and design v2 with changes explicitly motivated by v1 evidence.
