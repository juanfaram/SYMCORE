# Goodhart incident 001 — prefix-sum baseline

An initial Clang -O3 benchmark appeared ~10x faster than GCC because the harness only made the final prefix value observable. Under that weakened contract, intermediate output stores could be eliminated.

Status: **REJECTED MEASUREMENT**.

Correction: force materialization of every prefix output in the C harness before using the result as an external baseline.

Lesson: benchmark observables are part of Γ. A faster measurement under a weaker Γ is not an optimization result.
