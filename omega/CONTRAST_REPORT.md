# OMEGA CONTRAST REPORT

Generated from real accessible evidence. Canonical Open Discovery-01 run is not modified.

## Central claims C1-C8

C1 VERIFIED
Claim: Discovery-01 reaches O(N+T).
Source: run 36875825075; job 110414864764; artifact 11169296408; claim discovery-01-stateful-statistic.
Raw evidence: complexity.discovered="O(N+T)", lower_bound="Omega(N+T)"; speedups 537.5046x, 1503.1407x, 3209.4847x.
Caveat: the benchmark does not use a field named bad; correctness evidence is pytest differential/adversarial/property tests, not a sorting-style bad counter.

C2 VERIFIED
Claim: SC-AB compiled path loses to torch.compile baseline by about 4.3% in ratio terms.
Source: run 36873942476; job 110408453426; artifact 11168620932.
Raw: compiled_ns=13233753.5; omega_compiled_ns=13826546.5; ratio=0.9571264596.
Note: saying "4.5% slower" depends on denominator; direct time overhead is ~4.48%, speed ratio deficit is ~4.29%.

C3 VERIFIED
Claim: membership reformulation reaches specialist expected class and approximately specialist performance at N=Q=10000.
Source: run 36874597130; job 110410672633; artifact 11167724568.
Raw: omega_ns=657503; hostile_ns=657823; omega_vs_hostile=1.00048669; both expected O(N+Q).

C4 VERIFIED
Claim: Omega detects 3/3 constructed traps; naive detects 0/3.
Source: run 36873529422; job 110407040026; artifact 11168307397.
Raw: omega_detected=3; naive_detected=0.

C5 VERIFIED
Claim: four non-canonical Open Discovery-01 reruns were caused by documentation pushes matching workflow paths.
Source: Actions branch listing + workflow file.
Runs: 36878369687, 36878416485, 36878424124, 36879069017.
Raw facts: event=push; commit titles are NOVELTY_PROTOCOL, FAILURE_ANALYSIS_PROTOCOL, STOP_CERTIFICATE_TEMPLATE, GITHUB_EVIDENCE_SNAPSHOT; workflow paths include omega/open_discovery01/**.
Classification: ORCHESTRATION_LEAK.
Canonical evidence remains run 36877418452 + SHA 6bb1c1738c9f34d28b842ee647da5a97a108541b.

C6 VERIFIED WITH CONNECTOR SCOPE
Claim: only juanfaram/SYMCORE is installed/authorized in the GitHub connector.
Source: search_installed_repositories_streaming returned exactly one repository.
Scope caveat: this proves connector-visible installation state, not that the user owns no other GitHub repositories.

C7 VERIFIED
Claim: omega/knowledge/patterns.jsonl is empty on omega/symcore/v0.2-deep.
Source: direct fetch_file.
Raw content: empty.

C8 VERIFIED
Claim: NetworkX issue #4935 describes the quotient_graph edge-iteration reformulation.
External source: https://github.com/networkx/networkx/issues/4935
Raw substance: maintainer states current code loops over block/node pairs and proposes going through edges of G and connecting partitions; later proposal explicitly targets O(|E|).
Decision: NetworkX quotient_graph is REDISCOVERY control, not Discovery-02 novelty target.

Summary:
VERIFIED: 8/8 (C6 scoped to connector visibility; C1 wording corrected regarding "bad").

## K1-K8 reclassification

Rule: PATTERN requires >=2 independent evidence sources/cases.

K1 PATTERN
Hostile baseline can reverse verdict.
Evidence A: SC-AB vs eager = 2.1032686x.
Evidence B: same candidate vs torch.compile = 0.9571265x.
Both arise in run 36873942476 but are distinct baselines within one experiment. Treat as two contrasts, not two independent domains. Status PATTERN_LOCAL, not cross-domain.

K2 HYPOTHESIS
Output equivalence != optimization activated.
Evidence: epsilon=0 false-green in SYMCORE.
No second independent case found.

K3 HYPOTHESIS
Model != measurement.
Evidence: energy 1-1/r classified MODEL_NOT_MEASURED.
No second independent case found.

K4 HYPOTHESIS
Benefit depends on regime -> guard/dispatcher.
Evidence: SYMCORE positive and negative regimes.
These are multiple regimes but one mechanism/domain; no second independent case/domain found.

K5 PATTERN
Class-changing reformulation dominates local tuning.
Evidence A: membership linear->set, up to 4905.07x vs linear.
Evidence B: Discovery-01 recompute->sufficient statistic, up to 3209.48x vs hostile.
Independent problem instances.

K6 HYPOTHESIS
Non-observable state is prime elimination territory.
Evidence: Discovery-01 removal of materialized list.
No second independent case found.

K7 PATTERN
Absence in frozen oracles != world novelty.
Evidence A: Discovery-01 PARTIAL_RELATIVE despite oracle absence.
Evidence B: NetworkX quotient_graph rejected because external issue already contains the reformulation.
Independent internal/external cases.

K8 HYPOTHESIS
Orchestration can violate freeze semantics.
Evidence: Open Discovery-01 documentation-triggered reruns.
No second independent orchestration incident found.

Counts:
PATTERN/PATTERN_LOCAL: 3
HYPOTHESIS: 5
UNVERIFIED: 0

## Additional sources searched

Internal:
- v0.2 Claim Ledger.
- Discovery-01 Claim Ledger.
- Raw logs for policy, ablation, 3A, 3B, Discovery-01.
- Actions run listing for Open Discovery-01.
- Active workflow path filters.
- patterns.jsonl.
- historical capability gate.

External:
- NetworkX issue #4935 and current quotient_graph documentation.

No evidence was found to elevate K2/K3/K4/K6/K8 to cross-case PATTERN status.

## Was "reality found defects architecture did not foresee" true?

PARTIAL.

1. ORCHESTRATION_LEAK:
Not explicitly prevented by the frozen-run protocols. The later truth hierarchy pinned evidence to run ID/SHA, which limits evidentiary damage but does not prevent reruns. Therefore the operational defect was not prevented.

2. patterns.jsonl empty:
The architecture required knowledge persistence conceptually, but no synchronization/invariant enforced that extracted patterns be written to patterns.jsonl. Defect not prevented operationally.

3. CAPABILITY_GATE.yaml stale:
The truth hierarchy and versioned-evidence model partially mitigate this because an old branch file should not override newer run evidence. However the system lacked an explicit derived-current-state mechanism, so stale snapshot risk remained.

Overall claim: PARTIAL/STRONGLY SUPPORTED, not absolute VERIFIED.

## Current experiment state

Canonical Open Discovery-01:
repo=juanfaram/SYMCORE
branch=omega/open-discovery-01-sort17
run=36877418452
sha=6bb1c1738c9f34d28b842ee647da5a97a108541b
status at last check=in_progress

No non-canonical rerun counts as v1 evidence.
No CANDIDATE_NOVEL or NO_DISCOVERY is declared.
