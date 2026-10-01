# PASO 1 — NECESSITY GUARD · STOP CERTIFICATE

Run ID: 36873094651
SHA under test: ae228ecae863794d700256442ab7769c680b7bca
Workflow: omega-policy-gate.yml
Job: policy / 110405548836
Tests: 4/4 PASS
Raw pytest: "4 passed in 1.73s"
Exit: 0 (GitHub Actions step conclusion: success)

Policy:
- L < 1536 -> BYPASS
- expected r < 4 -> BYPASS
- otherwise -> ATTEMPT

Known intentional false negative:
- L=2048, r=2 had ~1.133x median validation speedup in Lab 01 but is rejected by r<4.
- Rationale: v0.2 prioritizes avoiding known regression regions over capturing every modest speedup.

Evidence level: E2
Status: SUPPORTED

Residual risk:
- Only four policy decision cases are directly tested here.
- Threshold provenance is CPU + MockTransformer + exact synthetic periodic Lab 01 validation.
- No claim for natural data, real models, GPU, or untested regimes.
- A finer guard based on cheap density/ratio estimation remains a live hypothesis.

Infrastructure genealogy:
1. Run 36872555309 / SHA 7389caf... -> INFRA_FAILURE: torch absent; tests not collected.
2. Run 36872837118 / SHA 9371328... -> INFRA_FAILURE: numpy absent; tests not collected.
3. Run 36872968456 / SHA aedd8ba... -> policy executed successfully but file exposed 2 pytest functions containing 4 assertions.
4. Run 36873094651 / SHA ae228eca... -> four explicit cases; 4/4 PASS.

Claim Ledger:
- policy-necessity-guard-v0.2 -> SUPPORTED / E2.

Judgment:
PASO 1 = CLOSED.

Next:
PASO 2 — Ablation Omega vs naive may now be OPENED.
