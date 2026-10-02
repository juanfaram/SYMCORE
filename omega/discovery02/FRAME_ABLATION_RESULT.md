# Frame Ablation Result

Run: 36892203427
SHA: 771eb9b80e557fd1b915845089b7147968ef0e15
Frozen interpretations commit: 2d4536562212beb9037c0311bdd3ea5622ea7cf0

Medians:
B builtin anext = 344.980620 ns/item
W minimal Python wrapper = 276.105754 ns/item
D direct __anext__ = 253.211202 ns/item

Deltas:
B-W = 68.874866 ns
W-D = 22.894552 ns
(B-W)/(B-D) = 0.7505

Frozen classification: F2 (operationally B >> W approx D).
Allowed conclusion: generic Python wrapper/frame does not reproduce the full builtin cost; builtin-specific work/path remains.
Forbidden conclusion: frame hypothesis globally false.

Belief update:
largest uncertainty is which component/path specific to builtin anext accounts for the extra cost.
Next cheapest discriminator: same-build progressive component ablation (builtin, lookup wrapper, default/sentinel wrapper, simple wrapper, direct).

State=(PASS,PASS,KNOWLEDGE_GAIN,L2 diagnostic).
