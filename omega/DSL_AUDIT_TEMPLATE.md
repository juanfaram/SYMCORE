# Future Discovery DSL Audit Template

Use only if a future problem requires a DSL. Open Discovery-01 searches complete algorithms and does not use this DSL.

Checks:
1. Neutrality — no operator names or semantics encode the expected solution.
2. Domain leakage — prohibit target-domain words that reveal the transformation.
3. Composition leakage — combinations of operators must not make the target trivial by construction.
4. Expressivity — document what solution classes are reachable and unreachable.
5. Oracle separation — no oracle output may enter operator design after freeze.
6. Provenance — every operator predates the run or is frozen before seeing results.
7. Ablation — measure whether removing any operator destroys discovery.
8. Counterfactual — test the same DSL on unrelated problems to estimate solution-specific bias.

Outcome: PASS / PARTIAL / FAIL.
A FAIL invalidates novelty claims from that DSL-driven run.
