# S1 candidate designs

## SC-A — tensorized exact mirror/periodic detection
Goal: remove Python loops over token pairs/period comparisons for each window.
Constraint: preserve detector precedence mirror -> periodic -> scale and epsilon semantics.
Design: compute vector differences for mirror and candidate periods with torch operations; reduce norms/masks on device.
Gate: exact contract tests + adversarial tests must pass before benchmarking.
Risk: changing norm/reduction ordering near epsilon boundary.

## SC-B — scale detection without per-element scalar extraction
Goal: eliminate repeated .item() calls and host-device synchronization risk.
Design: compute first-half and second-half norms as tensors, mask valid base norms, compute median ratio on device, verify scaled residual vectorized; convert only final decision/metadata when necessary.
Gate: same detected type/factor within declared numerical tolerance on validation fixtures.
Risk: median/floating reduction differences; CUDA evidence required before any GPU claim.

## SC-AB
Composition of A+B. Must be measured independently; no additive-gain assumption.
