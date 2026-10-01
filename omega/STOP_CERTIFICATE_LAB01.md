# Ω Stop Certificate — SYMCORE Lab 01

## Identity
- OmegaVersion: 0.1-lab
- Baseline branch: omega/symcore/lab-01
- Best candidate: SC-AB (omega/symcore/candidate-AB)
- Scope: GitHub Actions CPU; PyTorch MockTransformer; synthetic symmetry workloads.

## Correctness
- 17/17 contract + adversarial tests pass on candidate A, B and AB.
- Finite floating inputs are enforced.
- Exact-symmetry tests require actual compression, not merely round-trip equality.
- Correctness status: SUPPORTED (testing), not formally PROVED.

## Causal evidence
- S0 profile: detect_symmetry ~0.814s / compress ~0.841s.
- SC-A profile mean: ~105.1 ms/compress.
- SC-B profile mean: ~94.0 ms/compress.
- SC-AB profile mean: ~34.1 ms/compress.
- Approximate profile improvement SC-AB vs S0: ~4.94x.

## End-to-end regime map
Validation seeds: 1101–1104; B=1; D=64; periodic exact inputs.
| L | r | median speedup | range |
|---:|---:|---:|---:|
|1536|8|1.362x|1.273–1.387x|
|1536|4|1.077x|1.049–1.185x|
|1536|2|0.795x|0.775–0.803x|
|2048|8|2.076x|2.020–2.097x|
|2048|4|1.607x|1.593–1.624x|
|2048|2|1.133x|1.124–1.150x|

## Judgment
- Small/low-structure workloads: REFUTED as an end-to-end optimization.
- Large/high-compression periodic synthetic workloads: SUPPORTED at E3 within scope.
- General claim for real transformers, natural data, CUDA/GPU or energy: UNKNOWN.
- Energy-saving claim remains MODEL_NOT_MEASURED.

## Residual risks
- Synthetic periodic distribution may not represent production workloads.
- GitHub hosted CPU runners are not controlled benchmarking hardware.
- No blind split has been consumed.
- No GPU evidence.
- No direct energy measurement.
- No formal equivalence proof.

## Stop reason
Lab-01 has answered its primary question: the original implementation was dominated by detection overhead; vectorizing mirror/periodic + scale detection creates a real crossover region, but benefit depends strongly on sequence length and compression ratio. Further work should move to a new OmegaVersion and real workload validation rather than tune further on this validation set.
