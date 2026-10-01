# H4b CERTIFICATION RESULT

Certification run: 36890577993
Matrix SHA: 9119a1dbd47ee6f918794c3dfd8701dc13bccebc
Frozen interpretations commit: b40f4c991344f4eca2cc39811a8c49cb258e8e60

Same-workflow medians:
parent = 199.029754 ns/item
affected = 288.862116 ns/item
H4b = 291.352786 ns/item

All three: test_asyncgen + test_builtin + test_coroutines PASS.

Frozen classification: CASE A.
H4b is approximately affected and slightly slower.

delta_recovery = -0.862235%
delta_parent = -46.386548%
RRR = -0.027726

State(E4b-cert)=(PASS,PASS,NONE,L3).
Functional outcome: NONE / candidate rejected.
Epistemic outcome: KNOWLEDGE_GAIN.
Search result 137.674 ns did not generalize to controlled same-workflow certification.
H4 causal hypothesis remains UNKNOWN; T4b does not recover the regression under certification.

P-001 outcome: WRONG_GAIN.
Semantic risk prediction did not materialize in tested suites; gain prediction was wrong.
TransferGain: UNMEASURED because Policy_K=Policy_0.

Belief update:
- H3 lookup-form cause: REFUTED_IN_SCOPE.
- H4b private C slot helper while retaining Python frame: REFUTED_AS_EFFECTIVE_TRANSFORMATION.
- Largest remaining unknown: cost of entering/executing the Python anext frame itself vs semantics tied to that frame/default/introspection.
Next experiment should discriminate frame/call overhead at minimum cost before another optimization patch.
