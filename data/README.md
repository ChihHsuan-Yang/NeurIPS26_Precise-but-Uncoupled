# data/

| Path | What |
|---|---|
| `omni-math-2-filtered/` | the paper's corpus: 4,181 problems, ten tiers |
| `derived/` | the pinned Track A intermediate (65.6 MB transitions CSV) |
| `example/` | small stratified slices for smoke tests |
| `manifests/` | sha256 digests for everything above |
| `hf/` | *(created by `make fetch-data`, gitignored)* the downloaded dataset |
| `traces/` | *(optional, gitignored)* a per-tier trace tree, if you rebuild one |

Full documentation, schema, licensing and regeneration instructions:
[../docs/DATA.md](../docs/DATA.md).

Verify everything: `make verify-checksums`.
