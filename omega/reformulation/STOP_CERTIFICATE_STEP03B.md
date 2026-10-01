# PASO 3B — REFORMULATION CALIBRATION · STOP CERTIFICATE

Run ID: 36874597130
Job ID: 110410672633
SHA under test: db5eb57e6646ebc81ce1c1ad9cf684527c7ae01e
Workflow: omega-reformulation-3b.yml
Exit: 0
Verification: 2 pytest suites PASS, including 500 deterministic randomized cases.

Contract:
For each query, return whether an equal hashable value occurs in values, preserving query order.

Algorithms:
- Original: repeated linear scan, O(N*Q).
- Omega: set(values) + membership queries, expected O(N+Q).
- Hostile specialist: direct idiomatic set solution, expected O(N+Q).
- Batch lower bound: Omega(N+Q) to consume both collections under this contract.

Measured medians:
N=Q=100:    Omega/linear 73.67x; Omega/hostile 0.956x.
N=Q=1000:   Omega/linear 815.23x; Omega/hostile 0.908x.
N=Q=5000:   Omega/linear 2778.34x; Omega/hostile 0.959x.
N=Q=10000:  Omega/linear 4905.07x; Omega/hostile 1.0005x.

Judgment:
- Reformulation pipeline calibration PASSES.
- Omega reaches the same expected algorithmic class as the specialist.
- Omega does not claim superiority over the specialist.
- Novelty claim: NONE. set-based membership is a known reformulation.
- Evidence: E2, not E3; this run is one CI environment and the transformation was manually instantiated as a calibration target.

Capability demonstrated:
Contract -> Representation Lift -> Algorithm Change -> Verification -> Hostile Baseline -> Lower-Bound Reasoning -> Stop.

PASO 3B = CLOSED.

Next gate:
Select ONE genuine discovery candidate where strong oracles do not already implement/propose the reformulation.
