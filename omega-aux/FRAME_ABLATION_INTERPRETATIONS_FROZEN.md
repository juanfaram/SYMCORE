# FRAME ABLATION INTERPRETATIONS — FROZEN BEFORE RESULT READ

B=builtin anext; W=minimal Python wrapper; D=direct __anext__.

F1: W ≈ B >> D
Allowed: Python wrapper/frame overhead is a material explanation.
Next: identify which frame observables contract requires.
Forbidden: jump directly to "accelerate the frame".

F2: B >> W ≈ D
Allowed: generic Python frame does not reproduce builtin cost; builtin-specific work remains.
Next: differentiate builtin vs minimal wrapper.
Forbidden: general fast-path patch without discrimination.

F3: B ≈ W ≈ D
Allowed: ablation is inconclusive for the observed regression.
Next: redesign experiment.
Forbidden: frame hypothesis is false.

F4: B ≈ W >> D
Allowed: entering Python is material; separate frame/call/wrapper contributions.
Next: ask whether eliminating that cost can preserve contract.

UNKNOWN != FALSE. UNKNOWN != FAILURE.
