# Discovery-01 contract: evolving collection statistic

Input:
- initial finite sequence S0 of signed integers;
- stream of operations, each either ADD(x) or REMOVE(x), where REMOVE is valid only for a present occurrence.

Observable:
- after every operation, output the exact sum of squared values currently present:
  F(S_t) = sum(v*v for v in S_t).

Baseline:
- materialize the collection and recompute F by scanning the full current collection after every operation.

Constraints:
- exact Python integer semantics (no overflow);
- duplicate values are allowed;
- operation order is observable;
- output after every operation is observable.

Optimization objective:
- minimize total end-to-end time for processing the complete operation stream, including any setup/state costs.

No target reformulation is named in the search input.
