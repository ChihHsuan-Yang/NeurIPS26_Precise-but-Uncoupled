# Source of truth

When two numbers disagree, this page says which one wins. It exists because the
paper, the post-rebuttal analysis, and this release were produced at different
times, and a reader deserves an explicit ordering rather than having to guess.

---

## The paper of record

**arXiv:2607.15388v1**, 16 July 2026, cs.AI, DOI 10.48550/arXiv.2607.15388.
v1 is the only version. Status: **Preprint** — no acceptance decision is claimed.

Exact title:

> Precise but Uncoupled: Reviewer Precision Does Not Guarantee Critique Uptake in
> Multi-Agent Math Reasoning

Ten authors, in order: Chih-Hsuan Yang (corresponding), Jingyan Jiang,
Vikram Vasudevan, Cheng-Hau Yang, Huihuo Zheng, Le Chen, Eliu A. Huerta,
Venkatram Vishwanath, Ian T. Foster, Rajeev Thakur.
**Authors 1 and 4 are different people** — Chih-Hsuan Yang and Cheng-Hau Yang.

No BibTeX entry existed anywhere for this paper; the one in
[README](../README.md) and [CITATION.cff](../CITATION.cff) was authored for this
release from fields read off the live arXiv record.

---

## Precedence when sources disagree

1. **This release's reproducible artifacts** — anything you can regenerate and
   checksum with `make reproduce-analysis` / `make verify-checksums`.
2. **[LIMITATIONS.md](LIMITATIONS.md)** — the withdrawal record. It **overrides the
   paper** wherever they conflict. The paper prints cost numbers that a
   post-submission audit invalidated; the paper is wrong there and this document
   says so.
3. **arXiv v1** for everything else: the claims, the methods, the accuracy results.
4. Drafts, revision branches and working notes: **not sources**. One superseded
   revision draft in the authors' working tree has a different title, different
   scope, and asserts a claim that was retired. Nothing in this release derives
   from it.

---

## Where the correction layer lives

The paper body has not changed since before the arXiv posting; all post-submission
findings live outside it. This release carries them as a separate layer rather than
as edits to the paper:

| Finding | Where |
|---|---|
| Token/call/cost values are invalid and withdrawn | [LIMITATIONS.md](LIMITATIONS.md) §1 |
| Claims the release does not make | [LIMITATIONS.md](LIMITATIONS.md) §2 |
| Figure 4 coupling panel not reproducible; use Table 14's values | [LIMITATIONS.md](LIMITATIONS.md) §3.1 |
| Approval gate changes the measured transition regime | [METRICS.md](METRICS.md) |
| The gate cannot be isolated with the available toggle | [LIMITATIONS.md](LIMITATIONS.md) §2 |
| Compute matching is infeasible in this harness | [LIMITATIONS.md](LIMITATIONS.md) §1 |
| Run-to-run variability, quantified | [REPRODUCIBILITY.md](REPRODUCIBILITY.md) §2.3 |
| EMB's gain does not reproduce on the frozen subset | [LIMITATIONS.md](LIMITATIONS.md) §2 |
| The 5-dataset × 2-actor robustness matrix (absent from v1) | [PAPER_TO_ARTIFACT_MAP.md](PAPER_TO_ARTIFACT_MAP.md) |
| The resource-accounting fix | [LIMITATIONS.md](LIMITATIONS.md) §1 |

**The arXiv v1 abstract and the compiled PDF's abstract differ slightly** — the
posted metadata abstract is a shortened rewrite. Where a quotation matters, quote the
arXiv metadata abstract: it is the abstract of the artifact a reader can download,
and it is the more conservative of the two.

---

## Code of record

| Component | Status |
|---|---|
| **Analysis** (the process metrics) | **byte-identical** to the submitted run's base commit, verified per file by blob hash |
| **Protocol runtime** (Track B) | vendored at a **later** commit than the submitted run. Re-running reproduces the design, not the submitted outputs |
| **2×5 pipeline** | recovered verbatim from the post-rebuttal analysis, with one documented path patch |

Details, including every release-time edit: [PROVENANCE.md](PROVENANCE.md).

---

## Scope of record

The **paper** is Omni-MATH only — N=4,181, ten tiers, actor and evaluator both
`openai/gpt-oss-120b`, temperature 0, no seed.

The **5-dataset × 2-actor matrix** is post-rebuttal robustness evidence. It is
labelled as such in every document here, and `configs/rebuttal/` is kept separate
from `configs/paper/` for that reason.

Presenting the matrix as the paper's benchmark set would misstate the work.

---

## Model of record

**There is none.** No weights were trained, fine-tuned or released. Verified by
exhaustive filesystem walk and all-refs history search. Every model used is a public
third-party checkpoint, obtained and served by the user.
