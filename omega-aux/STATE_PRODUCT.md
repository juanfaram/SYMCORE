# State Product

State=(Execution, Semantics, Effect, Verification).
Execution: PASS|FAIL|UNKNOWN. Semantics: PASS|FAIL|UNKNOWN. Effect: NONE|LOW|MID|HIGH|UNKNOWN. Verification: NONE|L0|L1|L2|L3|L4.

UNKNOWN never means FALSE.

H4 E: (FAIL, UNKNOWN, UNKNOWN, NONE) because patch did not apply.
H3 E: (PASS, PASS, LOW(+0.069%), L2).
