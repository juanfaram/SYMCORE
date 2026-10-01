# Ω optimizer layer

Ω is kept separate from SYMCORE's existing compression core.

Current milestone: **cost function + corpus**.

Primary objective: minimize p50 latency. Secondary objectives: peak memory and source bytes. Every measured result must include MAD, repetitions, scale and environment.

Corpus v1 intentionally spans five structures: reduction, map/filter fusion, membership/data-representation change, scan/prefix sum, and recurrence.

Next milestone is to run this corpus, derive per-case lower bounds where justified, and only then activate external oracle execution.
