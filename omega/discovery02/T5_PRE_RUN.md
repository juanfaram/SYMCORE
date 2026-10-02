# T5 Pre-Run Questions

Q1 What do we know?
- Same-build component ablation: builtin 258.828 ns, lookup wrapper 257.165, default/sentinel wrapper 245.504, simple wrapper 214.117, direct 199.048.
- Upstream new traceback/call-graph tests exercise anext(..., default), not the one-argument fast path.
- Public signature must be representable.

Q2 Largest uncertainty?
Whether a nargs dispatcher can preserve the Python frame/coroutine semantics for the 2-argument default path while restoring direct C slot dispatch for 1 argument.

Q3 Cheapest discriminating experiment?
Implement one C builtin with Argument Clinic-compatible nargs dispatch: direct slot for 1 arg; call a private frozen-Python helper for 2 args. Run targeted tests then benchmark.

Q4 Which result changes which decision?
Semantic fail -> reject T5.
Semantic pass + <10% same-run Search gain -> low value/stop or next hypothesis.
Semantic pass + >=10% -> Certification matrix.

Q5 Evidence sufficient to stop?
For candidate promotion: targeted tests PASS and >=10% L2 gain. For final claim: L3/L4 same-run certification.
