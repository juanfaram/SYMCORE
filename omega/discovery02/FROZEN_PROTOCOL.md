# DISCOVERY-02 — CPython anext() Functional Freeze

External repo: python/cpython
Issue: #158407
Affected commit: 0a6c1ed34118b091230ee38fc047bcc2df8e5c5e
Pre-regression parent: 8df6f077115484361b40952751fb7d703843c192
Upstream fix at freeze: NONE OBSERVED in issue development section.

Observed external regression:
- macOS arm64: 141.8 -> 228.4 ns/item (61.1%).
- AWS Lambda optimized x86_64: control-adjusted +39.8% and +37.0%.
- direct generator.__anext__ did not regress; explicit builtin anext did.

Contract:
Preserve Python-visible anext semantics at affected commit, including default handling,
exceptions, async iterator behavior, traceback/introspection changes introduced by #157362,
and relevant upstream tests.

Functional objective:
Recover as much of the lost explicit-anext performance as possible without reverting
the intended observable semantics of the Python implementation.

Frozen baseline phase:
1. build parent SHA;
2. build affected SHA;
3. run identical benchmark on same runner;
4. preserve raw samples and build metadata.

Search hierarchy after baseline passes:
Necessity -> call-path representation -> algorithm/control flow -> C/Python boundary
-> specialization/cache -> compiler -> micro.

Success:
MEJORA_VERIFICADA requires semantic tests PASS and robust improvement vs affected,
with comparison to parent baseline. Novelty is optional metadata.

No benchmark-only special casing.
