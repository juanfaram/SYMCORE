# Iteration 3 — sanitized external baseline

## Measurement status

Run 36860665746 is the first admissible PolyBench run for Iteration 3.

All six kernel/compiler pairs escalated from MINI to SMALL because MINI was below the timer-resolution gate. SMALL passed:
- MAD / p50 <= 5%
- p50 >= 100 x 1us timer resolution

Hardware and compiler metadata are stored in the workflow artifact.

## SMALL p50 ± MAD (seconds)

| Kernel | GCC -O3 | Clang -O3 |
|---|---:|---:|
| 2mm | 0.000240 ± 0.000003 | 0.000261 ± 0.000002 |
| ATAX | 0.000143 ± 0.000001 | 0.000126 ± 0.000001 |
| BiCG | 0.000015 ± 0.000000 | 0.000015 ± 0.000000 |

BiCG MAD=0 is acceptable under the additional resolution gate because p50=15us is NOT >=100us; therefore this row is still instrument-resolution limited and remains NON-CONCLUSIVE under the current policy.

## Interpretation

- 2mm: GCC faster in this run.
- ATAX: Clang faster in this run.
- BiCG: unresolved at current timer resolution.
- These are general GCC/Clang -O3 baselines, not full state-of-the-art specialist baselines.
- No Ω contender exists yet for the three PolyBench kernels, so Ω/GCC and Ω/Clang ratios are N/A, not zero and not one.
- 2mm class-optimality claim is only within the classical dense formulation; subcubic matrix multiplication is outside the explored model.

## Iteration-3 closure

Iteration 3 remains open because the required 8-case table needs Ω measurements for comparable rows, and BiCG remains timer-resolution limited. The next repair inside Iteration 3 is a nanosecond-resolution harness for BiCG and explicit N/A cells for absent Ω contenders rather than fabricated ratios.
