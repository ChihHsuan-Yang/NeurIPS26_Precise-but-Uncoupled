# results/

| Path | What |
|---|---|
| `derived_tables/` | the **released reference tables**, pinned by checksum |
| `figure_data/` | per-tier data behind the paper's figures (`make reproduce-figures`) |
| `derived_tables/_regenerated/` | *(gitignored)* your own regenerated output |
| `live/` | *(gitignored)* output of your own Track B runs |

The distinction matters: `derived_tables/*.csv` are the artifacts the paper's numbers
come from and their digests are asserted in
[../docs/PAPER_TO_ARTIFACT_MAP.md](../docs/PAPER_TO_ARTIFACT_MAP.md). Anything under
`_regenerated/` or `live/` is **your** output and is never shipped, so one reader's
run can never be mistaken for reference data.

`make reproduce-analysis` writes to `_regenerated/` and compares against the
reference; it never overwrites it.

## The headline artifact

`derived_tables/matrix_2x5_precise_uncoupled_strict.csv`
sha256 `6c77c01e6a02889cbf69b77d3f33d0f8e5784a150d3d1b29d9311b06435daef2`

Reviewer precision, strict uptake and verified repair for each of 10 dataset × actor
cells. See [../docs/METRICS.md](../docs/METRICS.md) for the definitions and the two
methodological guardrails the generating code enforces.
