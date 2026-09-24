# FINAL RELEASE REPORT — Precise but Uncoupled
Date: 2026-09-24T04:00:16Z

## Release Verdict
PASS WITH DOCUMENTED LIMITATIONS

Every mandatory artifact is live, publicly reachable, and independently verified. The remaining
items are an author-only external action (arXiv v2) and honestly-stated non-verifications.

## Live Artifacts
- Paper    : https://arxiv.org/abs/2607.15388  (v1, 16 Jul 2026, cs.AI, DOI 10.48550/arXiv.2607.15388)
- GitHub   : https://github.com/ChihHsuan-Yang/NeurIPS26_Precise-but-Uncoupled   (public, 200)
- Website  : https://chihhsuan-yang.github.io/NeurIPS26_Precise-but-Uncoupled/   (built, 200)
- Dataset  : https://huggingface.co/datasets/AgentsSci/NeurIPS26_Precise-but-Uncoupled (public, 200, VIEWER LIVE: preview/search/filter enabled, 6 configs)
- Model Card: Not applicable. No project-trained model exists — verified absent by exhaustive
  filesystem walk AND all-refs git history search, not by a glob.

## Frozen Revisions
- Code SHA           : 4f0fa045be0aee8f728361d254782eeaf140ff47   (tag v1.0.0, 265 files)
- Website SHA        : 61879376097253b39304d268364b2f28268a7581   (branch gh-pages, 12 files)
- Dataset revision   : 588dc89e1b12aae041cbe015a37c0081c6bd58d6   (111 files, 325.3 MB)
- Paper source       : RebuttalOverleafClone 41e2b1558ba8835f3f500388d67e34563764e337 (claims)
                       CouplingOverleafClone b32e7d0e051038ec4c22df75e8e6e609bf5e3b41 (metadata)
- Submitted base     : be9c47c90c95b44d60e2b3bdad49d68b11557039
- Accounting patch   : b4a2db6ee53811b1a578972421da6a9cc2a18286
- Pinned intermediate: sha256 7441a770e590da04cb0accbb381d9a012e905e93b1d7120f2c04b1b53e21023c
- Headline table     : sha256 6c77c01e6a02889cbf69b77d3f33d0f8e5784a150d3d1b29d9311b06435daef2

## Verification
- Tests                : 43/43 pass (clean venv). validate: 13/13 PASS.
- FRESH-CLONE REPRO    : cloned the PUBLIC repo to an empty dir, ran make reproduce-analysis
                         -> PASS, byte-identical to 6c77c01e...
- Track A robustness   : table byte-identical in THREE environments (py3.9.6/pandas2.3.3,
                         py3.13.0/pandas3.0.3, original) and END-TO-END from public upstream
                         (34,446 trajectories -> 95,799 transitions -> identical summary).
- Dataset self-contained: verifier copied ONLY the 110 staged files to an isolated dir with no
                         network and reproduced 20/20 cells.
- Dataset integrity    : 109/109 checksums OK, 0 FAILED; full manifest coverage (no unlisted
                         files); all 40 trace shards record-level identical to upstream.
- Joins                : 100.00% on all six ID relations in scope (26,551/26,551), zero dup keys.
- Link audit           : all four surfaces cross-link (arXiv / GitHub / Website / HF) — verified
                         anonymously against the LIVE pages.
- Security scan        : zero private paths, credentials, tokens, endpoints on every surface.
                         Every scan carried a positive control; claim-critical scans carried a
                         planted negative proving the checker fires.
- License audit        : composite. license: other + LICENSE/NOTICE/LICENSES + per-source table.
                         The inherited placeholder license: mit did NOT survive.
- CFF / Croissant      : CFF 1.2.0 valid; croissant conformsTo set, 12 recordSets, 251 fields,
                         mlcroissant PASSED.
- Mobile/visual QA     : 390px no horizontal scroll, 16px gutters; 0 unreachable scroll panes;
                         0 images without alt; contrast >= 4.5:1.

## Scientific Guardrails
SUPPORTED (verified end-to-end by the orchestrator, not accepted on report):
  reviewer DETECTION quality and successful critique UPTAKE/REPAIR are empirically separable.
  PER precision > Broadcast in 10/10 dataset x actor cells; within PER precision > verified
  strict repair in 9/10 (exception JEEBench/Gemma-4, only n=2 evaluable); within Broadcast 10/10.
OUT OF SCOPE / NEVER CLAIMED:
  universal PER-vs-Broadcast ranking (Gemma-3 actors REVERSE it); approval-gate causality;
  semantic correctness of critique text; ALL legacy absolute token/call values; generalization
  beyond tested model families; any project-trained model; any NeurIPS acceptance.
RETIRED and absent from every surface except as a labelled retraction with its refutation:
  "PER has lower strict uptake in all settings" — measured 0/10 uptake, 3/10 repair. It was
  never in the submitted paper or arXiv v1, so no correction of record is owed.
SCOPE DISCIPLINE: the paper is Omni-MATH-only (N=4,181, temperature 0, actor and evaluator both
  openai/gpt-oss-120b). The five-dataset x two-actor matrix is post-rebuttal ROBUSTNESS evidence
  and is labelled as such on every surface.
COST: wall_time_seconds valid for all 53,224 matched trajectories; token/call NULL for exactly
  the 13,306 PER rows by construction; evaluator_calls treated INVALID pending re-derivation,
  with the audit disagreement documented rather than silently resolved.

## Original Source Preservation
AgentVerse /Users/bellayang/Documents/2026/multi_agents_trace/AgentVerse
  branch codex/dhd-protocol  (before == after)
  HEAD   b4a2db6ee53811b1a578972421da6a9cc2a18286  (before == after)
  porcelain 53 (before == after);  stash 1 entry (before == after)
  No tracked file added, modified, renamed, or deleted. No checkout/reset/stash/commit/push.
  HONEST EXCEPTION, not hidden: two files under .pytest_cache/ have session-window mtimes
  (18:48:59) because a sub-agent ran pytest with AgentVerse as CWD. That directory PRE-EXISTED
  (created 2026-07-17) and is gitignored (.gitignore:55), so git state is byte-identical to
  baseline. The claim "zero bytes were written under AgentVerse" would be FALSE and is not made.
note repo /Users/bellayang/Documents/note
  Files modified under Neurips-Coupling / the two Overleaf clones: 0.
  HEAD moved e759fe79 -> 882ec0f3 during the session. NOT OURS: git log over that range shows
  every changed path under ICLR-Receiver_Conditional_Communication_Policies, ZERO under
  Neurips-Coupling. We ran no mutating git command in this repo. All four extraction pins
  (05a343a7, 8cfd1766, 09932d23, df89bded) still resolve; the critical artifact still hashes to
  7441a770... The supportable claim is "no Neurips-Coupling path was modified", not "HEAD is
  unchanged".
Paper clones: all four re-verified at their audit HEADs. No .git directory was ever copied from
  a paper clone (three contain embedded Overleaf credentials).
Figure sources: all three website-sourced PDFs still match their audit sha256s.
Upstream HF dataset: CONTENT AND HISTORY UNCHANGED — HEAD still c36465ddb7c9, lastModified
  2026-08-09, no commit from this session. See UNRESOLVED item 2 on its visibility.
Nothing was deleted, moved, or overwritten in any original source. No force-push anywhere; the
  GitHub repo was empty (0 refs) before our first push, so publication was a create.

## Remaining Actions (genuine, not polish)
1. AUTHOR ONLY — submit arXiv v2. Exact text in docs/ARXIV_UPDATE_INSTRUCTIONS.md. Until then
   the paper of record links only to the HF Space and the cross-link matrix is INCOMPLETE on the
   arXiv surface. Three of four surfaces cross-link today.
2. AUTHOR DECISION — AgentsSci/scientific-agent-protocol-traces is now PUBLIC; it returned 401
   earlier this session. Its content and history are provably unchanged and I issued no
   visibility call against it, but I did not capture a 401 immediately before the flip, so I
   cannot prove I did not cause it and do not claim to. If unintended, set it back to private:
   nothing in this release depends on it (we ship the pinned intermediate precisely so Track A
   never needs upstream).
3. Consider git-lfs for data/derived/full_release_symmetric_transitions.csv (62.6 MiB; GitHub
   warns above 50 MB, blocks above 100 MB). It pushed fine and Track A depends on it being a
   plain file in the clone.

## Not Verified (stated, not softened)
Live Pages behaviour in Safari/Firefox; real touch hardware; screen readers; dataset revision
53a3a8d4 was never re-run (only c36465dd); make fetch-data has now been exercised against the
public dataset but not from a machine with no HF credentials at all.
