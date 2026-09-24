# Contributing

This repository is the **archival artifact** for a specific paper. That shapes what
kinds of change are welcome.

## What is very welcome

* **Reproduction reports** — you ran Track A and it passed, or it didn't. Either is
  useful. Include your OS, Python version, pandas version, and the digest you got.
* **Bugs in the reproduction path** — a script that fails from a fresh clone, a
  missing dependency, a path that only works on the authors' machine. These are real
  defects; please open an issue with the exact command and full output.
* **Documentation that is wrong or unclear**, especially in
  [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and
  [docs/PAPER_TO_ARTIFACT_MAP.md](docs/PAPER_TO_ARTIFACT_MAP.md).
* **Independent re-analysis** of the released data, including analysis that disagrees
  with ours. Please say which artifacts and digests you used.

## What we will decline, and why

* **Changes to the analysis modules under `src/agentverse/metrics/`.** These are
  byte-identical to the code that produced the published numbers. Changing them
  breaks the one property that makes this artifact verifiable. If you find a genuine
  bug there, please open an **issue** — it is a finding about the paper, not a pull
  request.
* **"Cleaning up" the vendored runtime.** Divergence from the original is a cost, not
  an improvement. Every existing deviation is deliberate and documented in
  [docs/PROVENANCE.md](docs/PROVENANCE.md).
* **Updating the reference artifacts** in `results/derived_tables/` or
  `data/derived/`. They are pinned by checksum on purpose.
* **New claims in the docs** that the released data does not support. See
  [docs/LIMITATIONS.md](docs/LIMITATIONS.md) for what the evidence does and does not
  license — in particular, no universal protocol ranking, no causal claim about the
  approval gate, no semantic claim about critique text, and no resurrection of the
  withdrawn cost numbers.

## Before opening a pull request

```bash
make test         # 43 tests: pipeline + release integrity guards
make validate     # anonymity, digests, structure
make reproduce-analysis   # must still print PASS
```

`make validate` will reject private absolute paths, credential-shaped strings, stray
build artifacts, and the retired claim. These are not stylistic checks — they are
what keeps the published artifact safe to distribute.

## Reporting something sensitive

If you find a credential, a private path, or anything else that should not have been
published, please follow [SECURITY.md](SECURITY.md) rather than opening a public
issue.

## Citation

If this artifact is useful in your work, please cite the paper — see the BibTeX in
the [README](README.md).
