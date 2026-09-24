# Precise but Uncoupled

### Reviewer Precision Does Not Guarantee Critique Uptake in Multi-Agent Math Reasoning

**Accepted to NeurIPS 2026 — Main Track**

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS%202026-Main%20Track%20Accepted-2E7D32)](https://neurips.cc/)
[![Project page](https://img.shields.io/badge/Project-Page-1f6f8b)](https://chihhsuan-yang.github.io/NeurIPS26_Precise-but-Uncoupled/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.15388-b31b1b)](https://arxiv.org/abs/2607.15388)
[![PDF](https://img.shields.io/badge/Paper-PDF-333333)](https://arxiv.org/pdf/2607.15388)
[![HF dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Card-ffcc4d)](https://huggingface.co/datasets/AgentsSci/NeurIPS26_Precise-but-Uncoupled)
[![Track A](https://img.shields.io/badge/Track%20A-offline%20%C2%B7%20byte--identical-2E7D32)](docs/REPRODUCIBILITY.md)
[![Licence](https://img.shields.io/badge/Licence-Apache--2.0%20%C2%B7%20data%20composite-green)](LICENSE)

Chih-Hsuan Yang¹\*, Jingyan Jiang¹, Vikram Vasudevan², Cheng-Hau Yang¹, Huihuo Zheng¹, Le Chen¹, Eliu A. Huerta¹ ³, Venkatram Vishwanath¹, Ian T. Foster¹ ³, Rajeev Thakur¹
¹ Argonne National Laboratory · ² Oregon State University · ³ University of Chicago · \* corresponding author

---

If you put a **reviewer** in a multi-agent system, and that reviewer is good at
spotting wrong answers, does the system solve more problems? **Not on its own.**
Being right about the error and *acting* on it are separable. On 4,181
Omni-MATH problems the planner–executor–reviewer pipeline has the *more precise*
reviewer, yet its critique is far less likely to change the answer the protocol
carries forward — and it solves fewer problems than plain broadcast discussion.

## Key finding

On 4,181 verifier-grounded Omni-MATH problems with `gpt-oss-120b` as both actor
and evaluator, at temperature 0:

| Metric | PER | Broadcast | Reading |
|---|--:|--:|---|
| Reviewer precision | **0.861** | 0.644 | PER is the **better detector** |
| CouplingRate | 0.336 | **0.935** | useful critique changes the next candidate |
| Reviewer-guided repair | 0.051 | **0.286** | …and that change is correct |
| FinalPassRate | 85.2% | **89.2%** | the outcome that follows |

Detection quality and critique uptake are **empirically separable**. A protocol
can look strong at spotting errors and still fail to solve more problems.

## Download everything here

No need to scroll. Every artifact is one click from this table.

| Resource | What it is | Get it |
|---|---|---|
| **Paper** | NeurIPS 2026 Main Track · 48 pp, 20 figures, 25 tables | [![arXiv](https://img.shields.io/badge/arXiv-abs-b31b1b)](https://arxiv.org/abs/2607.15388) [![PDF](https://img.shields.io/badge/-PDF-333333)](https://arxiv.org/pdf/2607.15388) |
| **Project website** | Story, figures, results, citation | [![Page](https://img.shields.io/badge/Project-Page-1f6f8b)](https://chihhsuan-yang.github.io/NeurIPS26_Precise-but-Uncoupled/) |
| **Dataset** | 53,224 trajectories · 95,799 review transitions · 325 MB | [![HF](https://img.shields.io/badge/%F0%9F%A4%97-Dataset%20card-ffcc4d)](https://huggingface.co/datasets/AgentsSci/NeurIPS26_Precise-but-Uncoupled) [![Viewer](https://img.shields.io/badge/-Browse-4c9aff)](https://huggingface.co/datasets/AgentsSci/NeurIPS26_Precise-but-Uncoupled/viewer) |
| **Headline table input** | Pinned 65.6 MB transitions CSV, Track A input | [![Download](https://img.shields.io/badge/-Download-2E7D32)](https://huggingface.co/datasets/AgentsSci/NeurIPS26_Precise-but-Uncoupled/resolve/main/derived/full_release_symmetric_transitions.csv) |
| **2×5 result table** | The published process table, 5 KB | [![Download](https://img.shields.io/badge/-Download-2E7D32)](https://huggingface.co/datasets/AgentsSci/NeurIPS26_Precise-but-Uncoupled/resolve/main/derived/matrix_2x5_precise_uncoupled_strict.csv) |
| **Release tag** | Frozen v1.0.0 | [![Tag](https://img.shields.io/badge/-v1.0.0-6f42c1)](https://github.com/ChihHsuan-Yang/NeurIPS26_Precise-but-Uncoupled/releases/tag/v1.0.0) |
| **Model** | *None. No model was trained for this paper.* | — |

### Models used (dependencies, not our artifacts)

| Model | Role | Licence | Card |
|---|---|---|---|
| `openai/gpt-oss-120b` | Actor **and** evaluator, all primary results | Apache-2.0 | [![HF](https://img.shields.io/badge/%F0%9F%A4%97-Model-ffcc4d)](https://huggingface.co/openai/gpt-oss-120b) |
| `google/gemma-3-27b-it` | Appendix second family, *n* = 835 | Gemma Terms | [![HF](https://img.shields.io/badge/%F0%9F%A4%97-Model-ffcc4d)](https://huggingface.co/google/gemma-3-27b-it) |

### Benchmarks (5 × 2 × 4 robustness matrix)

Omni-MATH is the paper's evidence. The other four are **post-rebuttal
robustness**, not the main claim.

| Benchmark | Slice | *n* | Role | Licence | Source |
|---|---|--:|---|---|---|
| **Omni-MATH 2** | `competition_math_4181` | 4,181 | **Primary** | Apache-2.0 | [![HF](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-ffcc4d)](https://huggingface.co/datasets/martheballon/Omni-MATH-2) |
| JEEBench | `text_only` | 515 | Robustness | MIT | [![Repo](https://img.shields.io/badge/-GitHub-181717)](https://github.com/dair-iitd/jeebench) |
| SciBench | `text_only` | 574 | Robustness | MIT | [![Repo](https://img.shields.io/badge/-GitHub-181717)](https://github.com/mandyyyyii/scibench) |
| LAB-Bench | `llm_strict` | 741 | Robustness | CC-BY-SA-4.0 · do-not-train | [![HF](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-ffcc4d)](https://huggingface.co/datasets/futurehouse/lab-bench) |
| MaScQA | `text_only` | 642 | Robustness | **CC-BY-NC-SA-4.0** · NonCommercial | [![Repo](https://img.shields.io/badge/-GitHub-181717)](https://github.com/M3RG-IITD/MaScQA) |

### Documentation

| Document | Read it for |
|---|---|
| [Release artifacts](docs/RELEASE_ARTIFACTS.md) | Every public URL and the revision to cite |
| [Reproducibility](docs/REPRODUCIBILITY.md) | What reproduces offline, and what does not |
| [Paper → artifact map](docs/PAPER_TO_ARTIFACT_MAP.md) | Which script and file back each table |
| [Metrics](docs/METRICS.md) | Precision vs uptake vs repair, defined |
| [Limitations](docs/LIMITATIONS.md) | **Read before quoting any number** |
| [Provenance](docs/PROVENANCE.md) | Source commits, checksums, contact audit |
| [arXiv v2 instructions](docs/ARXIV_UPDATE_INSTRUCTIONS.md) | The one open author action |

## Two tracks

| | Track A — offline | Track B — live |
|---|---|---|
| Needs a model endpoint | **No** | Yes, your own |
| Determinism | **Byte-identical** | Varies; temperature 0 but no seed |
| Runtime | ~1 minute | Hours, and it costs tokens |
| Entry point | `make reproduce-analysis` | `make smoke-live` |

## Quickstart

```bash
git clone https://github.com/ChihHsuan-Yang/NeurIPS26_Precise-but-Uncoupled.git
cd NeurIPS26_Precise-but-Uncoupled
pip install -r requirements.txt

make reproduce-analysis   # Track A: regenerate the 2x5 table, no model calls
make test                 # 43 tests
make validate             # 13 release guards
```

`make reproduce-analysis` ends by comparing the regenerated table against
`sha256 6c77c01e…daef2` and fails loudly on any mismatch. It reproduces
byte-identically under Python 3.9.6 / pandas 2.3.3 and Python 3.13 / pandas
3.0.3, and end-to-end from the public dataset.

To run the protocols against your own OpenAI-compatible service:

```bash
export OPENAI_BASE_URL="https://your-endpoint.example/v1"
export OPENAI_API_KEY="your-key"
make smoke-live
```

## Load the dataset

```python
from datasets import load_dataset

repo = "AgentsSci/NeurIPS26_Precise-but-Uncoupled"
outcomes  = load_dataset(repo, "outcomes",         split="train")
coupling  = load_dataset(repo, "coupling_metrics", split="train")
problems  = load_dataset(repo, "problems",         split="train")

# the paper's own setting
omni = outcomes.filter(lambda r: r["benchmark_id"] == "omnimath2")
print(len(omni), omni.column_names)
```

Six viewer-ready configs: `outcomes`, `coupling_metrics`, `problems`,
`aggregate_metrics`, `evaluator_replay`, `trace_index`. No token needed — the
dataset is public. Whole-repo download:

```bash
hf download AgentsSci/NeurIPS26_Precise-but-Uncoupled --repo-type dataset --local-dir ./data
```

> **`before_correct` and `after_correct` are tri-state** (True / False / null).
> Null means the frozen evaluator never labelled that side — **not** "incorrect".
> Filter on explicit values; never coerce null to False.

## Which numbers are invalid — read before quoting any cost figure

Legacy per-problem token and model-call values were **cumulative within
worker**, not per problem. Exact recovery is impossible. The specific
absolute token totals and verifier-cost thresholds printed in the preprint
are withdrawn and must not be quoted; see
[docs/LIMITATIONS.md](docs/LIMITATIONS.md) for the enumerated list and why
each one is unrecoverable.

| Field | Status |
|---|---|
| `wall_time_seconds` | **Valid**, all 53,224 trajectories |
| tokens / calls, non-PER protocols | **Valid**, 39,918 rows |
| tokens / calls, PER | **NULL by construction** — 13,306 rows, not recoverable |
| `evaluator_calls` | **Treated as invalid** pending re-derivation |
| Correctness, Pass@1, evaluator replay, transition labels | **Unaffected** |

The accounting fix and its regression test ship in this repo, separate from the
legacy results.

## Scope discipline

The paper is **Omni-MATH only**: *n* = 4,181, ten tiers, `gpt-oss-120b` as both
actor and evaluator, temperature 0. The five-dataset × two-actor matrix is
**post-rebuttal robustness evidence** and is labelled as such everywhere.

Not supported by this release: a universal PER-vs-Broadcast ranking — Gemma-3
actors **reverse** it (PER 65.6% vs Broadcast 58.7%); any causal claim about the
approval gate; semantic correctness of critique text; generalization beyond the
tested families.

## Licence

Code: [Apache-2.0](LICENSE). Data: **composite and inherited** — MaScQA is
CC-BY-NC-SA-4.0 (**NonCommercial**) and LAB-Bench is CC-BY-SA-4.0 with an
upstream do-not-train request. See [`NOTICE`](NOTICE).

## Citation

```bibtex
@misc{yang2026preciseuncoupled,
  title        = {{Precise but Uncoupled: Reviewer Precision Does Not Guarantee
                  Critique Uptake in Multi-Agent Math Reasoning}},
  author       = {Yang, Chih-Hsuan and Jiang, Jingyan and Vasudevan, Vikram and
                  Yang, Cheng-Hau and Zheng, Huihuo and Chen, Le and
                  Huerta, Eliu A. and Vishwanath, Venkatram and Foster, Ian T. and
                  Thakur, Rajeev},
  year         = {2026},
  note         = {Accepted to NeurIPS 2026 (Main Track)},
  eprint       = {2607.15388},
  archivePrefix= {arXiv},
  primaryClass = {cs.AI},
  doi          = {10.48550/arXiv.2607.15388},
  url          = {https://arxiv.org/abs/2607.15388}
}
```

## Acknowledgments

This research used resources of the Argonne Leadership Computing Facility, a
U.S. DOE Office of Science user facility at Argonne National Laboratory,
operated under Contract No. DE-AC02-06CH11357.

We thank the authors of Omni-MATH, JEEBench, SciBench, LAB-Bench and MaScQA,
whose benchmarks this study measures against. See [`NOTICE`](NOTICE).

## Contact

Chih-Hsuan (Bella) Yang — bellayang@anl.gov
