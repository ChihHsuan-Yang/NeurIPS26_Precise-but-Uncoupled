# Provenance

Where every file came from, what was changed for the release, and what was left
out. The organising question: **can a reader tell whether the code they are
reading is the code that produced the paper's numbers?**

Short answer: for the **analysis** code, yes, byte-for-byte. For the **protocol**
runtime, no — and that is stated wherever it matters rather than glossed.

---

## 1. Sources

| Component | Origin |
|---|---|
| Analysis modules, paper configs, derived report CSVs | the authors' experiment repository at commit `b4a2db6e` |
| The submitted run's base | commit `be9c47c9` in the same repository |
| 2x5 symmetric-transition pipeline | the post-rebuttal analysis branch of the authors' notes repository |
| Upstream project this derives from | [OpenBMB/AgentVerse](https://github.com/OpenBMB/AgentVerse), Apache-2.0 |
| Benchmark corpus | Omni-MATH 2 (filtered), MIT |

## 2. The central provenance result

`code/coupling/` in the source repository is a **byte-identical extraction of the
submitted base `be9c47c9`**. Verified per file by git blob hash, independently
during this build:

```
21/21 analysis modules   HEAD:code/coupling/analysis/<f>  ==  be9c47c9:agentverse/metrics/<f>
 5/5  scripts            HEAD:code/coupling/scripts/<f>   ==  be9c47c9:scripts/<f>
 2/2  job scripts        HEAD:code/coupling/jobs/<f>      ==  be9c47c9:jobs/<f>
```

So the analysis that produced the paper's process metrics is exactly what ships in
`src/agentverse/metrics/`.

### 2.1 The duplicate-module hazard, and how it was resolved

The source repository carried the same 21 analysis modules **twice**: in
`code/coupling/analysis/` and in `agentverse/metrics/`. Two consequences had to be
handled, not ignored:

**(a) The two copies had diverged.** Two files differed:

| File | Status |
|---|---|
| `main_paper/evaluator_labeling.py` | differed at HEAD |
| `diagnostics/labeling.py` | differed at HEAD |

At the paper's commit both pairs were **identical**; they diverged later under an
unrelated project. This release restores both to the `be9c47c9` blob, verified:

```
main_paper/evaluator_labeling.py  staged == be9c47c9 == code/coupling  (612701a6...)
diagnostics/labeling.py           staged == be9c47c9 == code/coupling  (9ca564ce...)
```

**(b) The extracted copy was dead code.** Every `__init__.py` under
`code/coupling/analysis/` re-exported *from* `agentverse.metrics.*`, and the `.pyc`
files under it recorded `agentverse/metrics/` as the compiled source path — i.e.
**the code that actually ran was `agentverse/metrics/`**, and the extracted tree
could not run standalone.

This release therefore ships **one definition per function**, in
`src/agentverse/metrics/`, and exposes it under paper-facing names through a
re-export facade at `src/precise_uncoupled/analysis/`. There is no second copy.
Two tests enforce this (`TestSingleDefinition`), and every analysis entry point is
import-tested (`TestImportability`).

## 3. Track B is not byte-exact — the honest version

While the analysis code matches `be9c47c9` exactly, the **protocol runtime does
not**. Relative to the submitted base, these differ at `b4a2db6e`:

| File | Δ lines |
|---|---|
| `environments/tasksolving_env/rules/per.py` | +184 / −8 |
| `environments/tasksolving_env/rules/broadcast_deliberation.py` | +98 / −10 |
| `agents/tasksolving_agent/deliberator.py` | +28 / −11 |
| `output_parser/output_parser.py` | +45 |
| `llms/openai.py` | +438 / −55 |
| `agentverse_command/benchmark.py` | +88 / −2 |
| `initialization.py`, `tasksolving.py`, `environments/base.py`, `agents/base.py`, and 4 others | smaller |

Most of these are additive hooks from a separate critique-intervention project.
Evidence that they are inert for this paper: the policy object defaults to
`enabled=False`, `from_config(None)` yields a disabled instance, `decide()` returns
immediately when disabled, and a grep across the paper config directories finds
**zero** references to any of the new keys.

**That is strong but not conclusive.** It shows the new code paths are not
*entered*; it does not prove the shared code they wrap is bit-identical. Hence the
label used throughout: Track B *re-runs the protocol at `b4a2db6`, which post-dates
the submitted run `be9c47c9`*. We do not claim a live re-run reproduces the
submitted outputs.

Two further notes on what was and was not taken:

* For `llms/openai.py` the **committed** blob at `b4a2db6e` was used, not the
  working-tree copy, which carried ~100 lines of uncommitted debug logging.
* The per-example resource-accounting fix (counter reset + regression tests) is
  part of `b4a2db6e` and is therefore present. It post-dates the runs whose cost
  numbers were withdrawn; see [LIMITATIONS.md](LIMITATIONS.md).

## 4. Release patches — every edit made to source files

Every edit is marked in the source with a `RELEASE PATCH` comment. None changes any
computation; all are path, portability or side-effect fixes.

| File | Patch |
|---|---|
| `process/aggregate_release_transitions.py` | default release root / output dir were an author-private cache; now repo-relative, `PU_RELEASE_ROOT` / `PU_OUTPUT_DIR` |
| `process/analyze_matrix_precise_uncoupled.py` | three defaults pointed at the authors' working directory; now repo-relative, `PU_TRANSITIONS` / `PU_OUTPUT_DIR` |
| `scripts/bootstrap_problem_clustered_metrics.py` | `--result_root` default was an absolute private path to a directory that no longer exists; now repo-relative, `PU_TRACE_ROOT` |
| `scripts/export_paper_per_problem_tables.py` | same defect, same fix |
| `jobs/aggregate_paper_tier_reports.sh` | conda bootstrap + absolute `REPO_ROOT` + renamed data dir; now derives its root from its own location |
| `jobs/rerun_paper_trace_analysis.sh` | as above, plus it sourced a private endpoint/token script that is not part of this release (and is not needed — it reads saved traces) |
| `llms/openai.py` | facility endpoint hostname and gateway URL path removed; workarounds now generic and opt-in. Import-time probe of `localhost:5000` made opt-in. Rate-monitor log file made opt-in |
| `logging.py` | wrote `logs/` **inside the package** on import; now defaults to the working directory and honours `AGENTVERSE_LOG_DIR` |
| `initialization.py` | BMTools import removed (dangling submodule, unused); `load_tools()` now raises a clear error instead of failing obscurely |
| `tests/test_*.py` (2 recovered files) | the path to the module under test was recomputed for the new layout; assertions unchanged |
| `data/omni-math-2-filtered/summary.json` | a `files` block held 12 absolute author paths; replaced with a repo-relative path and a note. Counts untouched |

Verification that the analysis logic survived: a full diff of
`analyze_matrix_precise_uncoupled.py` against the recovered original shows changes
confined to one added import and the argparse defaults — 298 → 311 lines, with no
edit inside any analysis function.

### 4.1 Recovered-file digests, as staged

| File | sha256 (first 16) |
|---|---|
| `analyze_matrix_precise_uncoupled.py` (as recovered) | `068bb2f16e2fd2a4` |
| `symmetric_transition_extractor.py` | `c6707314ada565b1` |
| `aggregate_release_transitions.py` (as recovered) | `0f87f4df30b7eba3` |
| `audit_release_sample.py` | `ad339c2581015156` |
| `full_release_symmetric_transitions.csv` | `7441a770e590da04` |
| `matrix_2x5_precise_uncoupled_strict.csv` | `6c77c01e6a028896` |

The two `.py` files marked "as recovered" carry a RELEASE PATCH in the shipped
copy; the digests above are the pre-patch originals, recorded so the patch can be
audited by diff.

## 5. What was excluded, and why

Excluding was preferred to guessing. Nothing below is needed for any paper claim.

### 5.1 Other research projects sharing the same repository

| Excluded | Reason |
|---|---|
| `metrics/critique_intervention/` | separate critique-intervention project |
| `metrics/local_credibility/`, `metrics/poll_speak/` | separate projects |
| `shapley/` and its CLI | separate attribution project |
| DHD protocol (`diverse_hypothesis` env/rule, `dhd_parsing`, hypothesizer/integrator agents) | first committed two months after the paper's analysis; different project |
| `configs/paper/{dhd,dhd_gemma4,lab_bench,science_qa}` | other benchmarks/projects, outside paper scope |
| `configs/paper/{per,broadcast}_hint_llm_metis/` | untracked, post-date the paper runs, differ from their parents only by a model-alias string |
| `configs/paper/*_pilot/`, `*_intervention/` | 100- and 500-problem subset configs; the paper's results are full-benchmark. Name collides with the separate intervention project, so excluded rather than guessed |
| `different_model_family/CRUX_README.md` | site-specific scheduler operations doc containing facility storage paths and a private endpoint URL |

### 5.2 Upstream AgentVerse code this paper does not use

| Excluded | Reason |
|---|---|
| the simulation half (`environments/simulation_env/`, `agents/simulation_agent/`, `simulation.py`) | no paper config uses a simulation `env_type`; pulls langchain and a tool stack |
| `gui.py`, `demo.py` | Gradio UIs |
| `memory_manipulator/reflection.py` | simulation-only; pulls scikit-learn + numpy. The `basic` manipulator the paper uses is kept |
| `executor/{tool_using,code_test,coverage_test}.py` | no paper config selects them; one pulled spaCy transitively, one hardcoded a local tool-server URL |
| `agentverse/tasks/` (upstream demo task bundle) | paper runs pass `configs/paper/<protocol>` directly |
| upstream demo dataloaders (commongen, humaneval, logic_grid, responsegen) | paper uses `omni-math` |
| `BMTools/` | tracked only as a dangling gitlink with no `.gitmodules`; unimportable, unused |
| `use_alcf.sh`, `inference_auth_token.py` | facility endpoint + Globus token helpers; replaced by generic env vars |

Modules that upstream `__init__` files still import were **kept** where they are
byte-identical to the submitted base and cheap (e.g. the manager agent), so the
vendored tree stays as close to the original as possible. Each removal is paired
with a `RELEASE NOTE` comment at the import site that referenced it.

### 5.3 Never present

* **Model checkpoints.** Verified absent by exhaustive filesystem walk *and* an
  all-refs git history search: no `.safetensors`/`.bin`/`.pt`/`.ckpt`/adapter files
  at any commit, no training code on the paper path. There is no model card because
  there is no model.
* **Credentials.** No key, token, cookie or `.env` at any point.
* **`.git` directories** from any source tree.

## 6. Anonymity and secret scan

`scripts/validate_release.py` and `tests/test_release_integrity.py` both scan the
tree for private absolute paths, credential-shaped strings, resolvable private
endpoints, the retired claim, and withdrawn cost values.

Both scanners carry **positive controls**: if a pattern stops matching its own
known-bad string, the check fails loudly rather than reporting a clean tree. Both
also assert they are reaching a plausible number of files, so a broken walker
cannot masquerade as a pass. The only files exempt from scanning are the two
scanner files themselves — because they necessarily contain the forbidden patterns
as controls — and a test asserts the exemption covers exactly those two files, so a
new file dropped into `scripts/` or `tests/` is still scanned.

Retained by deliberate decision: coarse facility and backend labels (e.g. `aurora`,
`crux`, `sophia`) where they appear as data values. They are public facility names,
already published, and load-bearing for the run-to-run variability analysis. They
are **not** hostnames. Resolvable endpoint URLs, IPs, ports, usernames, absolute
paths and tokens remain forbidden and are absent. The author e-mail
`bellayang@anl.gov` is retained: it is the corresponding-author address already
published on arXiv.

## Contact details and third-party addresses (audit, 2026-09-23)

A full sweep of every tracked file was run for e-mail-shaped strings.

**In code, scripts, configs, notebooks and the Makefile: zero.** No example
snippet, execution script, or usage instruction contains a personal address,
an internal path, or a credential. Verified over all tracked `*.py`, `*.sh`,
`*.yaml`, `*.yml`, `*.json`, `*.ipynb` and `Makefile` files.

**`bellayang@anl.gov` appears 7 times, all deliberate citation or contact
metadata**, never in runnable code:

| File | Role |
|---|---|
| `CITATION.cff` | author e-mail, required by the CFF schema for the corresponding author |
| `pyproject.toml` | package author metadata |
| `README.md` | corresponding-author line under the author list |
| `SECURITY.md` (x2) | where to report a vulnerability, plus the note explaining why it is retained |
| `CODE_OF_CONDUCT.md` | where to report a conduct concern |
| `docs/PROVENANCE.md` | this record |

These are kept intentionally. The address is already published on arXiv v1 and
on the paper's title page; a project with no reachable maintainer cannot receive
a security report or a correction. Replacing them with a placeholder would make
`SECURITY.md` and `CODE_OF_CONDUCT.md` non-functional and would break the CFF
citation record. The distinction that matters for this release is between a
*published contact address* (kept) and a *filesystem path containing a
username* (removed everywhere -- `validate_release.py` enforces zero hits for
the four absolute-path prefixes it defines, covering macOS home directories,
Linux home directories, and the two ALCF filesystem roots).

**Two third-party addresses appear in `data/omni-math-2-filtered/all.jsonl`**
(`vladimir.shelomovskii@gmail.com`, `orders@tomasdiaz.com`), in the
`equation_solution` field of exactly 2 of the 4,181 records. These are
Art-of-Problem-Solving contributor attributions carried verbatim in the
upstream Omni-MATH-2 benchmark text. They are **not ours to edit**: the file is
redistributed benchmark content under Apache-2.0, its digest is pinned in the
release manifest, and altering the solution text would corrupt the benchmark
and break checksum-verified reproduction. They are reported here rather than
silently modified.
