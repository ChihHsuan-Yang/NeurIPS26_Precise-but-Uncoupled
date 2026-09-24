# Data

What ships in this repository, what you download separately, and how to check both.

---

## 1. What ships here

| Path | Size | What |
|---|---|---|
| `data/omni-math-2-filtered/all.jsonl` | 7.6 MB | the paper's full corpus, N=4,181, ten tiers |
| `data/omni-math-2-filtered/summary.json` | 2 KB | tier counts and difficulty histogram |
| `data/derived/full_release_symmetric_transitions.csv` | 65.6 MB | the pinned Track A intermediate, 104,729 rows × 59 cols |
| `data/example/omni_math_example_20.jsonl` | small | 2 problems per tier, stratified — for smoke tests |
| `data/example/omni_math_example_3.jsonl` | small | 3 problems (tiers 1, 5, 8) — for `make smoke-live` |
| `data/manifests/*.sha256` | — | digests for everything above |

**Why the 65 MB intermediate ships.** So Track A works offline, in about a minute,
with no download and no account. Regenerating it from raw traces is possible
(§3) but not required.

### Record schema — `all.jsonl`

| Field | Meaning |
|---|---|
| `question` | problem statement (LaTeX) |
| `answer_number` | the evaluator-usable final answer |
| `equation_solution` | reference solution text |
| `difficulty_tier` | 1–10, the tier used throughout the paper |
| `difficulty`, `raw_difficulty` | the underlying continuous score |
| `domain`, `source` | provenance of the original problem |
| `dataset_name`, `judge_type` | fixed metadata |

---

## 2. The released dataset (downloaded separately)

```bash
make fetch-data     # or: bash scripts/fetch_data.sh
```

Target: **https://huggingface.co/datasets/AgentsSci/NeurIPS26_Precise-but-Uncoupled**

Lands in `data/hf/NeurIPS26_Precise-but-Uncoupled/` (gitignored). Contains the
trajectory and message parquet files the transition pipeline consumes, plus the
derived tables.

### A 401 before publication is expected

The dataset is built private and flipped public at the final release step. If you
are reading the published paper and `make fetch-data` still 401s, the flip has not
happened — please open a GitHub issue. `fetch_data.sh` detects this case and says so
rather than printing a raw stack trace.

### Do not chase the internal upstream

Provenance documents mention an internal dataset used during the authors' own
analysis. It is **private and returns 401 to anonymous users**. It is referenced for
traceability only, never as a download instruction. This release is deliberately
**self-contained**: everything Track A needs either ships here or comes from the
public dataset above.

---

## 3. Regenerating the transition intermediate (optional)

```bash
make fetch-data

PYTHONPATH=src/precise_uncoupled/process \
python3 src/precise_uncoupled/process/aggregate_release_transitions.py \
  --release-root data/hf/NeurIPS26_Precise-but-Uncoupled \
  --output-dir   results/derived_tables/_regenerated \
  --scope        all-per-broadcast
```

Reads `data/trajectories/*.parquet` and `data/messages/*.parquet` under the release
root, streaming messages in batches. Takes about a minute.

Paths are configurable without editing code:

| Variable | Meaning |
|---|---|
| `PU_RELEASE_ROOT` | where the downloaded dataset lives |
| `PU_OUTPUT_DIR` | where derived outputs are written |
| `PU_TRANSITIONS` | the transitions CSV that `reproduce-analysis` consumes |
| `PU_TRACE_ROOT` | root of a per-tier trace tree, for the report-building scripts |

**Expect the regenerated CSV to differ in bytes from the pinned one** — same 104,729
rows, same 59 columns, same header, small field-value differences from documented
label corrections. The *table* it produces is cell-for-cell identical, which is the
property the paper depends on. Verified: 34,446 trajectories → 95,799 transitions →
all ten rendered rows diff-clean.

A note on scope: the aggregator emits 26 summary cells because it covers every
PER/Broadcast cell present in the release. The paper's 2×5 table is the
10-cell subset selected by the dataset × actor filter in the analysis script.

---

## 4. Verifying what you have

```bash
make verify-checksums
```

Recomputes every digest in `data/manifests/` and compares. **A missing file is a
FAILURE, not a skip** — absence must never read as a pass. The headline 2×5 digest is
checked by name in addition to via the manifest.

The checker has been exercised against three failure modes to confirm it
discriminates: a one-byte corruption (caught), a deleted file (caught as MISSING),
and a corrupted headline table (caught). It returns PASS on a clean tree.

---

## 5. Licensing

The **corpus in this repository** derives from Omni-MATH / Omni-MATH-2, both listed
as MIT on their public dataset cards. Credit is due to both benchmarks' authors; see
[NOTICE](../NOTICE).

The **released Hugging Face dataset** is broader — it carries the 5-dataset
robustness matrix — and therefore inherits **more restrictive per-source terms**,
including:

* a **NonCommercial** source (MaScQA, CC-BY-NC-SA-4.0) — commercial use not permitted;
* a **ShareAlike, do-not-train** source (LAB-Bench, CC-BY-SA-4.0), whose contamination
  canary is preserved verbatim.

**Read the dataset card before redistributing or using it commercially.** This code
repository is Apache-2.0; the dataset's terms are not the same and are not weakened
by it.

---

## 6. Data validity — do not skip this

Some fields in the released per-row data are **invalid by construction** and are
nulled or marked accordingly: the token and model-call counters accumulated across
problems within a worker process. Accuracy and process metrics are unaffected; all
absolute token/cost values are withdrawn and cannot be recovered.

Full account: [LIMITATIONS.md](LIMITATIONS.md) §1. Read it before computing anything
resource-related from this data.
