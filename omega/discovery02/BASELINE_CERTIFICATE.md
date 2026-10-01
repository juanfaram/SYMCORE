# DISCOVERY-02 — BASELINE CERTIFICATE

Run: 36884671052
Lab SHA: 67d2dbf5f972a97f93ebd785f1661db1231587eb
Runner: Linux 6.17 Azure x86_64, GCC 13.3.0

CPython parent:
8df6f077115484361b40952751fb7d703843c192
anext median: 198.710224 ns/item
MAD: 0.79605
async_for median: 153.325728 ns/item
Tests: test_asyncgen PASS; test_builtin PASS.

CPython affected:
0a6c1ed34118b091230ee38fc047bcc2df8e5c5e
anext median: 337.957704 ns/item
MAD: 2.412302
async_for median: 167.604716 ns/item
Tests: test_asyncgen PASS; test_builtin PASS.

Local regression:
anext time increase = 70.0757%.
affected/parent = 1.700757x.
parent/affected speed ratio = 0.587974x.

Control:
async_for time increase = 9.3129%.

Judgment:
SCENARIO A — REGRESSION REPRODUCED.
The affected commit produces a large explicit-anext regression on the same runner while both targeted upstream test suites pass.

Search is authorized.

Functional target:
recover or exceed parent anext performance while preserving affected-commit semantics/introspection.

Evidence level: E2.
