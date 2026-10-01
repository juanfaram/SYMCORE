# Open Discovery-01 — Novelty Protocol

Trigger: a fixed <=70 comparator network returns bad=0 under the exact verifier.

Status on trigger: CANDIDATE_NOVEL only.

Mandatory sequence:
1. Independent verification in a separate process/environment.
2. Verify exact comparator count and channel validity.
3. Exhaustively verify all 2^17 binary inputs again.
4. Canonicalize network under obvious channel relabel/reversal symmetries and search for duplicates.
5. Re-check current Dobbelaere sorting-network compilation and SorterHunter outputs.
6. Search current literature/preprints for 17-channel <=70 comparator networks and equivalent constructions.
7. Record provenance and SHA of candidate before any further optimization.
8. External disclosure to independent sorting-network experts/maintainers.
9. Public artifact/preprint with verifier and candidate.

Promotion:
CANDIDATE_NOVEL -> EXTERNALLY_UNRECOGNIZED only after steps 1-6.
FULL/WORLD-NOVEL is never self-certified by Omega; it requires external scrutiny/publication.

If any duplicate is found, downgrade to REDISCOVERY and preserve the evidence.
