# figures/

This directory is intentionally near-empty, and that is a deliberate choice rather
than an omission.

**This release publishes figure INPUT DATA, not a re-rendering pipeline.** The data
is in [`../results/figure_data/`](../results/figure_data/) and is exactly
reproducible and checksummed (`make reproduce-figures`).

Two honest reasons:

1. The plotting script bundled with the paper source emits only **14 of the 20**
   figures — it does not emit main-paper Figure 2, among others. The complete
   1,539-line variant lives outside the analysis repository.
2. Even that complete variant is a *near*, not byte-exact, regenerator: four of its
   own outputs differ in byte size from the published PDFs.

Shipping a pipeline that quietly disagrees with the published figures would be worse
than shipping the data and saying so.

Rendered figures for the web are maintained on the project website:
https://chihhsuan-yang.github.io/NeurIPS26_Precise-but-Uncoupled/

Per-figure reproducibility status, including which figures are **withdrawn** and must
not be republished: [../docs/PAPER_TO_ARTIFACT_MAP.md](../docs/PAPER_TO_ARTIFACT_MAP.md).
