# Open Discovery-01 — bad>0 Analysis Protocol

Trigger: all four frozen shards finish with bad>0.

First judgment:
NO DISCOVERY WITHIN SEARCH SCOPE.

Do not modify v1 retrospectively.

Analyze, in order:
1. Final and minimum bad per shard.
2. Improvement trajectory per shard.
3. Number of exact candidate evaluations.
4. Acceptance/restart behavior.
5. Distance of every one-comparator deletion of incumbent.
6. Whether search converged to common basins.
7. Which mutation families generated improvements.
8. Evidence of plateaus/local minima.
9. Budget utilization and evaluations/second.
10. Only after 1-9, propose v2 changes.

Every v2 change must cite a v1 observation that motivates it.
New seeds/budget/operators create a new OmegaVersion/run.
