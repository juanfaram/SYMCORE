# Historical H/T/A/E + State Migration

## Open Discovery-01 v1
H: H-SORT17 — local mutation/annealing around 71-comparator incumbent may find <=70 network.
H status: REFUTED_WITHIN_FROZEN_SCOPE.
T: T-SORT17-v1 — deletion-seeded local mutation search.
A: search.py + exact verifier.
E: run 36877418452.
State: (PASS, PASS, NONE, L4-equivalent exact verifier).
Effect: no incumbent improvement; all shards bad=2.

## D2 Baseline
H: H-D2-REG — affected CPython commit materially regresses explicit anext.
H status: SUPPORTED.
T: measurement only.
A: frozen benchmark + exact SHAs.
E: run 36884671052.
State: (PASS, PASS, HIGH(regression +70.0757%), L2 baseline).
Verification: targeted upstream tests PASS.

## H4a
H: H4 — Python/C boundary/frame/call path contains dominant recoverable cost.
H status: UNKNOWN.
T: T4a — replace hot anext with direct C async-slot fast path.
T status: UNKNOWN.
A: A4a1 h4_fastpath.patch.
A status: FAIL(PATCH_DOES_NOT_APPLY).
E: run 36885401370.
State: (FAIL, UNKNOWN, UNKNOWN, NONE).
Rule: no semantic/performance inference.

## H3
H: H3 — explicit type/class lookup + unbound __anext__ call is a material cause of regression.
H status: REFUTED_IN_SCOPE.
T: T3 — bound async_iterator.__anext__() call while retaining Python function/default semantics.
T status: SUPPORTED_AS_EXECUTABLE.
A: apply_h3.py.
A status: PASS.
E: run 36886689330.
State: (PASS, PASS, LOW(+0.0688% vs affected), L2).
Affected baseline median: 337.957704 ns/item.
H3 median: 337.725374 ns/item.
Gain: (337.957704-337.725374)/337.957704 = 0.06875%.
Tests: test_asyncgen + test_builtin PASS.

## H4b prospective
H: H4 remains UNKNOWN.
T: T4b — retain Python anext frame/default logic but delegate special-method slot dispatch to private C helper.
A: apply_h4b.py.
E: queued after commit d959079525ef14984ada77349444cf3e6d7529d0.
State: (UNKNOWN, UNKNOWN, UNKNOWN, NONE) until run evidence.
