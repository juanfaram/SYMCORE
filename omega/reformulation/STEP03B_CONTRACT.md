# Step 3B contract — membership reformulation

Observable: for each query, return whether an equal value occurs in values, preserving query order.

Domain for calibration: hashable Python values with equality/hash consistency.

Baseline implementation: repeated linear scan per query.
Omega reformulation: build set(values), then query the set.
Hostile specialist: idiomatic hand-written set(values) solution.

Correctness: exact boolean-list equality.
Verification: fixed adversarial examples + 500 deterministic randomized cases.
Complexity claim:
- baseline O(N*Q) worst/expected scan work;
- hash reformulation expected O(N+Q), subject to normal hash-table assumptions;
- batch lower bound Omega(N+Q) under the contract when both collections must be consumed.

Novelty claim: NONE. This is a calibration case for the reformulation pipeline.
