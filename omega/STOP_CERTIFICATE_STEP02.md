# PASO 2 — ABLACIÓN Ω vs NAÏVE · STOP CERTIFICATE

Run ID: 36873529422
Job ID: 110407040026
SHA under test: fedbedd24e247df8c56f8990d26fe7c3f0dbb695
Workflow: omega-ablation.yml
Raw artifact: omega-step02-ablation / ID 11168307397
Exit: 0 (workflow step success)

Result:
- Omega detects: 3/3
- Naive detects: 0/3
- H1 ("naive detects the same problems") = REFUTED within this corpus.

Trap table:
1. epsilon=0 false-green: Omega YES / naive NO.
   Observed: roundtrip=true, compressed=false, detected=false.
2. forward-only benchmark: Omega YES / naive NO.
   Observed Lab-01 constants: baseline_forward=5.54ms, forward-only=1.385ms,
   actual end-to-end=568.864485ms, actual speedup=0.009738699x.
3. modeled energy as measurement: Omega YES / naive NO.
   Observed: reported model=0.75, direct energy measurement=null,
   classification=MODEL_NOT_MEASURED.

Evidence level: E2
Claim: ablation-omega-vs-naive-v0.2
Status: SUPPORTED

Residual risk / limitations:
- Corpus contains only three deliberately constructed traps based on already-known Lab-01 failures.
- This demonstrates discrimination by the encoded epistemic protocol, not automatic discovery of unknown failure modes.
- Forward-only fixture reuses Lab-01 measured constants instead of fresh concurrent timing.
- Naive arm is intentionally minimal by experimental definition; stronger competing audit protocols remain untested.

Judgment:
PASO 2 = CLOSED.

Next:
PASO 3 — hostile baseline (torch.compile or strongest executable equivalent) may now be OPENED.
