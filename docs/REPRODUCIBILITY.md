# Reproducibility

Two tracks. Track A reproduces the paper's numbers exactly and needs no model.
Track B re-runs the experiment against your own endpoint and will **not** match
exactly — by construction, not by defect.

---

## 0. Environment

```bash
git clone https://github.com/ChihHsuan-Yang/NeurIPS26_Precise-but-Uncoupled
cd NeurIPS26_Precise-but-Uncoupled
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Verified on CPython **3.9.6** and **3.13.0**, macOS and Linux. Track A needs only
`pandas`, `pyarrow`, `PyYAML`, `colorama`; the rest of `requirements.txt` is for
Track B. Nothing here needs a GPU, a scheduler, or a facility account.

`make` is optional throughout — every target is a script in `scripts/`. GNU Make
3.81 (what macOS ships) is supported; the Makefile avoids `.ONESHELL`, `!=`, and
grouped targets for that reason.

---

## 1. Track A — exact offline reproduction

### 1.1 The one-minute version

```bash
make smoke-test          # imports, checksums, 43 unit tests
make reproduce-analysis  # rebuild the headline table, verify sha256
```

`reproduce-analysis` prints **PASS** only if the regenerated table is byte-identical
to the released one:

```
2x5 table sha256
  expected : 6c77c01e6a02889cbf69b77d3f33d0f8e5784a150d3d1b29d9311b06435daef2
  actual   : 6c77c01e6a02889cbf69b77d3f33d0f8e5784a150d3d1b29d9311b06435daef2
reproduce-analysis: PASS
```

### 1.2 What "exact" means here, and how far it was tested

This is not a tolerance-based claim. The same digest was obtained in three
environments:

| Environment | Result |
|---|---|
| Python 3.9.6, pandas 2.3.3 | `6c77c01e…daef2` |
| Python 3.13.0, pandas 3.0.3 | `6c77c01e…daef2` |
| the original analysis environment | `6c77c01e…daef2` |

Two Python minor lines and two pandas *major* lines. That is why the README says
EXACT MATCH rather than "approximately reproduces".

### 1.3 The full chain, and which link is pinned

```
 public HF dataset (trajectories + messages, parquet)
   |
   |  aggregate_release_transitions.py        [layer 1 -> 2]
   v
 full_release_symmetric_transitions.csv       sha256 7441a770...  (SHIPPED, pinned)
   |
   |  analyze_matrix_precise_uncoupled.py     [layer 2 -> 3]
   v
 matrix_2x5_precise_uncoupled_strict.csv      sha256 6c77c01e...  (SHIPPED, reference)
```

`make reproduce-analysis` runs **layer 2 → 3** from the pinned intermediate that
ships in `data/derived/`. That is deliberate: it means Track A works offline, in
about a minute, with no download.

**Layer 1 → 2 has also been executed end to end**, against the public trace data:
34,446 trajectories → 95,799 transitions → a summary identical to the published
one, with all ten rendered table rows diff-clean. To repeat it yourself:

```bash
make fetch-data                      # download the released dataset
PYTHONPATH=src/precise_uncoupled/process \
python3 src/precise_uncoupled/process/aggregate_release_transitions.py \
  --release-root data/hf/NeurIPS26_Precise-but-Uncoupled \
  --output-dir   results/derived_tables/_regenerated \
  --scope        all-per-broadcast

PU_TRANSITIONS=results/derived_tables/_regenerated/full_release_symmetric_transitions.csv \
make reproduce-analysis
```

**Expect the regenerated intermediate to differ in bytes from the pinned one**
(same 104,729 rows, same 59 columns, same header; a small field-value delta from
documented label corrections and id normalisation). The *table* it produces is
cell-for-cell identical, which is the property that matters. `reproduce-analysis`
warns rather than fails when the input digest differs, for exactly this reason.

One honest residual: the end-to-end run was executed against the current public
revision. The earlier revision's link is evidenced by the byte-identical archived
intermediate rather than by a fresh aggregate run from that revision.

### 1.4 What the headline table supports

```
cells 10 | per_precision_above_broadcast 10 | per_precision_above_own_repair 9
         | broadcast_precision_above_own_repair 10
         | per_uptake_below_broadcast 0 | per_repair_below_broadcast 3
```

* reviewer precision (PER) > reviewer precision (Broadcast) in **10/10** cells;
* within PER, precision > verified repair in **9/10** — the exception is
  JEEBench/Gemma-4 with only **n=2** evaluable follow-ups;
* within Broadcast, precision > verified repair in **10/10**.

The last two counters are why the release makes **no** universal ordering claim:
measured over the same ten cells, PER's strict uptake is lower than Broadcast's in
**0** of them and its repair in only **3**.

Two cells are genuinely marginal and should not be presented as clean separations:
Omni-MATH/GPT-OSS repair 0.035912 vs 0.035955, and LAB-Bench/GPT-OSS 0.076923 vs
0.075377 — both differ in the fourth decimal. Neither affects the precision counts
above.

### 1.5 Other Track A outputs

```bash
make verify-checksums     # recompute every manifest digest
make reproduce-rebuttal   # post-rebuttal robustness tables (labelled as such)
make reproduce-figures    # per-tier figure input data
make validate             # pre-publication self-check of the tree
```

---

## 2. Track B — live re-execution

### 2.1 Point it at your endpoint

```bash
export OPENAI_BASE_URL="https://your-endpoint.example/v1"
export OPENAI_API_KEY="..."
make smoke-live
```

Any OpenAI-compatible server. The paper served the public open-weight checkpoint
`openai/gpt-oss-120b` (Apache-2.0) at **temperature 0, max_tokens 4096**, with
`evaluator: {type: llm}` — the same checkpoint acting as verifier, held fixed
across protocols.

No endpoint, hostname, key or token is embedded anywhere in this repository, and
no scheduler is required. Facility-specific workarounds that existed in the
research code were made generic and **opt-in**; left unset they are inert:

| Variable | Effect when unset (the default) |
|---|---|
| `AGENTVERSE_ENDPOINT_QUIRKS_HOST` | no endpoint-specific workarounds |
| `AGENTVERSE_MODEL_ALIASES` | model ids passed through verbatim |
| `AGENTVERSE_BARE_MODEL_ID` | provider prefix preserved |
| `AGENTVERSE_TOKEN_REFRESH_CMD` | a 401 is raised, not silently re-minted |
| `AGENTVERSE_RATE_LOG`, `AGENTVERSE_LOG_DIR` | no log files written into the tree |

### 2.2 Full scale

```bash
PYTHONPATH=src python3 -m agentverse_command.benchmark \
  --task         configs/paper/per_hint_llm \
  --dataset_path "$PWD/data/omni-math-2-filtered/all.jsonl" \
  --output_path  "$PWD/results/live/per_full" \
  --overwrite
```

Repeat for `baseline_llm`, `single_agent_hint_llm`, `broadcast_hint_llm`. This is
4,181 problems per protocol; the multi-agent arms issue on the order of a hundred
model calls per problem, so budget accordingly and shard by tier
(`--example_start` / `--example_end`) if your endpoint has concurrency limits.

**Using a different model:** `benchmark.py` has no `--model` flag — the actor and
evaluator ids live in the config's `model:` fields. Copy a config directory and
edit them there.

### 2.3 What will and will not reproduce

**Expect to vary.** There is no seed. Temperature 0 does not make a hosted endpoint
deterministic: its runtime is not frozen and it exposes no seed parameter, so two
executions of the same config are independent samples.

Measured on a frozen 195-problem stratified subset, two independent executions on
the same backend:

| Quantity | Run 1 | Run 2 | Range |
|---|---|---|---|
| PER FinalPass | 86.67% | 87.18% | **0.51 pp** (SD 0.36) |
| Broadcast FinalPass | 93.33% | 91.28% | **2.05 pp** (SD 1.45) |
| paired Broadcast − PER | +6.67 pp | +4.10 pp | 2.56 pp (SD 1.81) |

On a second backend the paired gap was **+6.67 pp** as well. So run-to-run
variation (≤ ~2 pp) is smaller than the Broadcast−PER effect (+4.10 to +6.67 pp) —
though one of the four paired intervals does cross zero, and a single small run
tells you nothing about accuracy.

The frozen subset itself is defined without reference to any outcome: membership is
`sha256("coupling-rebuttal-frozen-subset-v1|" + problem_id)`, 20 ids per tier
(15 for tier 10). No outcome, token count or label entered the selection.

**Expect not to reproduce exactly.** The protocol runtimes here are vendored at
AgentVerse `b4a2db6`, which **post-dates the submitted run's base `be9c47c9`**. The
*analysis* modules are byte-identical to `be9c47c9` (verified per-file by blob
hash); the *protocol* modules are not. The differences are additive hooks from a
separate project, disabled by default and never referenced by any paper config —
but "not entered" is weaker than "identical", so Track B is labelled as reproducing
the experiment's **design**, not the submitted run's outputs. See
[PROVENANCE.md](PROVENANCE.md).

### 2.4 What to compare

Compare **directions and separations**, not absolute percentages:

* does reviewer precision exceed reviewer-guided repair, within each protocol?
* is the coupling rate much lower in PER than in Broadcast?
* does the tier-4-onward gain pattern appear?

Those are the paper's claims. A ±2 pp shift in FinalPass is expected and is not a
failed reproduction.

---

## 3. If something fails

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md). The most common cause of an
unexpected Track A failure is a modified or partially-downloaded input — run
`make verify-checksums` first; it fails loudly on a missing file rather than
skipping it.
