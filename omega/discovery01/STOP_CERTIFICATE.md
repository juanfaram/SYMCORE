# DISCOVERY-01 · STOP CERTIFICATE

Status: CLOSED
Problem: evolving collection with exact sum-of-squares observable after each valid ADD/REMOVE.

Selection criteria:
1. Class gap: YES — repeated rescans vs per-update sufficient-statistic transition.
2. Frozen strong oracles: YES — CPython/runtime, frozen specialist catalog, recompute specialist.
3. Verifiable: YES — adversarial + property + >=1000 differential deterministic cases.
4. End-to-end measurable: YES — complete operation stream including initialization.
5. Lower bound: YES — Omega(N+T) to consume initial state and T operations.
6. Multiple plausible reformulations: YES — 12 Semantic Destruction operators, 1 proposal each.

Frozen before search:
- Contract, D_dev/D_validation/D_blind.
- Oracle proposals.
- Hostile baseline.
- Search budget: 12 operators x 1 proposal.
- Blind remained sealed.

Search result:
Representation: scalar sufficient statistic A.
Derived transition:
- ADD(x): A <- A + x*x
- REMOVE(x): A <- A - x*x
Output: emit A after each operation.
Derived from: factor_delta, introduce_summary, invert_update, derive_recurrence.

Necessity refinement:
First realization retained a list solely to perform REMOVE. Since the contract guarantees valid REMOVE and final state is not observable, this state was eliminated. Final realization reaches O(N+T).

Final evidence:
Run ID: 36875825075
Job ID: 110414864764
SHA: aed3b67cba7e466a881ab5455e0eb18e91989df3
Verification: 3 suites PASS, including 1000 deterministic differential cases.
Validation:
- N=1024,T=2048: 537.50x vs hostile; candidate p50 0.186 ms.
- N=4096,T=4096: 1503.14x vs hostile; candidate p50 0.453 ms.
- N=8192,T=8192: 3209.48x vs hostile; candidate p50 0.907 ms.
All benchmarks: 11 samples, median + MAD recorded.

Class:
Hostile/reference: O(sum_t |S_t|), approximately O(T*N).
Found: O(N+T).
Lower bound: Omega(N+T).
Class gap closed under declared contract.

Novelty:
PARTIAL / RELATIVE TO FROZEN ORACLES.
The reformulation was not present in the frozen oracle catalog or detector as a domain rule.
However, maintaining algebraic sufficient statistics under updates is known mathematics/computer science. No FULL or world-novelty claim is made.

Evidence level: E3 (held-out validation seeds, repeated samples, differential/adversarial/property verification).
Blind: NOT CONSUMED.

Limitations:
- Constructed problem.
- Frozen oracle set is not equivalent to global state of the art.
- Valid REMOVE is a contractual assumption.
- Python integers avoid overflow concerns.
- Blind still sealed.

Judgment:
Discovery-01 demonstrates that Omega can derive a class-changing reformulation relative to predeclared oracles, then use Necessity to eliminate residual non-observable state and reach the lower-bound class.
It does NOT yet demonstrate a world-novel algorithm.

Next:
Discovery-02 should use an externally sourced real problem, freeze real specialist/compiler oracles, and prevent project-designed structure from leaking the solution.
