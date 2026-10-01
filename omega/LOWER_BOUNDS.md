# Ω lower bounds v1

| case | lower bound | candidate | complexity gap | conclusion |
|---|---|---|---|---|
| reduction_sum | Ω(N) | Θ(N) | Θ(1) | class-optimal |
| map_filter_fusion | Ω(N) | Θ(N) | Θ(1) | class-optimal |
| repeated_membership | Ω(N+Q) | expected Θ(N+Q) | expected Θ(1) | expected class-optimal under hash assumptions |
| prefix_sum | Ω(N) | Θ(N) | Θ(1) | class-optimal |
| repeated_subproblem | Ω(1) trivial | Θ(N) | O(N) vs trivial bound | unresolved |

An asymptotic lower bound is not a numeric time lower bound. Numeric candidate/time-bound ratios remain UNKNOWN until a calibrated physical lower bound exists.