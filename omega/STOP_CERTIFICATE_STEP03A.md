# PASO 3A — HOSTILE BASELINE · STOP CERTIFICATE

Run ID: 36873942476
Job ID: 110408453426
SHA under test: 6f6efb158f1a38342e751155f62d4450e5c579fc
Workflow: omega-hostile-3a.yml
Artifact: omega-hostile-3a / ID 11168620932
Exit: 0

Scope:
GitHub Actions CPU, PyTorch, Mock MultiheadAttention, B=1, L=2048, D=64,
exact periodic input, measured compression ratio r=8.

Medians:
- eager baseline: 30.0066195 ms
- torch.compile baseline: 13.2337535 ms
- SC-AB + eager forward end-to-end: 14.266661 ms
- SC-AB + compiled forward end-to-end: 13.8265465 ms

Ratios:
- SC-AB eager vs eager baseline: 2.1032686x
- SC-AB eager vs compiled baseline: 0.9275999x
- SC-AB compiled vs compiled baseline: 0.9571265x

Judgment:
- SC-AB strongly beats the weak eager baseline in this run.
- SC-AB DOES NOT beat the hostile torch.compile baseline end-to-end.
- Even with compiled forward inside the Omega path, preprocessing/reconstruction overhead leaves it ~4.5% slower.
- Therefore no state-of-the-art win is claimed.

Evidence: E2
Claim: hostile-torch-compile-v0.2 / SUPPORTED

Residual risk:
single CPU runner, single synthetic regime, mock model, no natural data, no GPU.

Knowledge gained:
The optimization reserve is no longer mainly in the transformer forward.
To beat the hostile baseline, Omega must reduce/avoid preprocessing overhead,
change the representation/algorithm, or operate in a regime with greater amortization.

PASO 3A = CLOSED.
Next authorized gate: PASO 3B — Reformulation calibration case.
