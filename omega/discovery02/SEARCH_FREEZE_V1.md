# DISCOVERY-02 SEARCH FREEZE v1

Baseline certificate: omega/discovery02/BASELINE_CERTIFICATE.md
Affected CPython: 0a6c1ed34118b091230ee38fc047bcc2df8e5c5e
Parent target: 8df6f077115484361b40952751fb7d703843c192

Budget v1:
- 5 hypothesis families.
- one primary implementation candidate per family before branching.
- max 90 minutes CI per candidate.
- no full CPython-wide optimization; scope is anext call path.

VOC MODEL ranking (ordinal until calibrated):
H4 hybrid C/Python boundary: functional gain HIGH, info HIGH, semantic risk MEDIUM, cost HIGH.
H2 lazy/conditional observability: gain HIGH, info HIGH, risk HIGH, cost HIGH.
H3 cached/specialized dispatch: gain MEDIUM-HIGH, info MEDIUM, risk MEDIUM, cost MEDIUM.
H1 necessity/elision of Python-level operations: gain MEDIUM, info HIGH, risk MEDIUM, cost LOW-MEDIUM.
H5 guarded type specialization: gain MEDIUM, info MEDIUM, risk MEDIUM, cost MEDIUM.

Execution order v1: H1 analysis -> H4 candidate -> H3 -> H2 -> H5, updated only by measured evidence.

Cost funnel:
L0: static semantic/diff review; reject if affected semantics cannot be preserved.
L1: build candidate + targeted test_asyncgen/test_builtin.
L2: frozen anext benchmark; require >=10% improvement vs affected to continue.
L3: 9+ repeated benchmark samples + relevant tests; compare async_for control.
L4: compare against affected AND parent; candidate must not use benchmark-only special casing.

Success:
MEJORA_VERIFICADA if tests pass and anext median <= parent median within robust measurement, or if a meaningful improvement remains below parent but is explicitly certified as partial improvement (not H1 milestone).

Hard constraint:
Do not simply restore the old C implementation if it removes affected-commit traceback/introspection semantics.
