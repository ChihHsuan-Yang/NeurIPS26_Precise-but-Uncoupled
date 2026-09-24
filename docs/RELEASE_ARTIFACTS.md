# Release Artifacts — live links and pinned revisions

Single reference for every public surface of this project. Last verified
2026-09-24 (all links returned HTTP 200 to an anonymous client).

## Live links

| Artifact | URL |
|---|---|
| Paper (arXiv abstract) | https://arxiv.org/abs/2607.15388 |
| Paper (PDF) | https://arxiv.org/pdf/2607.15388 |
| Code repository | https://github.com/ChihHsuan-Yang/NeurIPS26_Precise-but-Uncoupled |
| Project website | https://chihhsuan-yang.github.io/NeurIPS26_Precise-but-Uncoupled/ |
| Dataset (Hugging Face) | https://huggingface.co/datasets/AgentsSci/NeurIPS26_Precise-but-Uncoupled |
| Release tag | https://github.com/ChihHsuan-Yang/NeurIPS26_Precise-but-Uncoupled/releases/tag/v1.0.0 |
| Model card | Not applicable — no project-trained model exists |

## Pinned revisions

| Component | Revision |
|---|---|
| Code (`main`) | `317bd7368eaddd360e0cca8a23a209e27492813d` |
| Website (`gh-pages`) | `61879376097253b39304d268364b2f28268a7581` |
| Dataset (`main`) | `588dc89e1b12aae041cbe015a37c0081c6bd58d6` |
| Paper | arXiv:2607.15388**v1**, 16 Jul 2026, cs.AI |
| DOI | 10.48550/arXiv.2607.15388 |

## Content pins used for reproduction

| Artifact | sha256 |
|---|---|
| `data/derived/full_release_symmetric_transitions.csv` | `7441a770e590da04cb0accbb381d9a012e905e93b1d7120f2c04b1b53e21023c` |
| 2x5 headline table (`matrix_2x5_precise_uncoupled_strict.csv`) | `6c77c01e6a02889cbf69b77d3f33d0f8e5784a150d3d1b29d9311b06435daef2` |

`make reproduce-analysis` regenerates the second from the first and fails loudly
if the digest does not match. It needs no model endpoint and no network.

## Provenance of the underlying experiments

| | |
|---|---|
| Submitted experiment base | `be9c47c90c95b44d60e2b3bdad49d68b11557039` |
| Resource-accounting patch | `b4a2db6ee53811b1a578972421da6a9cc2a18286` |

## Related but distinct

The Hugging Face Space https://huggingface.co/spaces/AgentsSci/scientific-agent-protocol-traces-site
is a broader multi-paper resource and is the URL printed in arXiv v1's Comments
field. It is **not** this paper's project website; the GitHub Pages URL above is.
See `docs/ARXIV_UPDATE_INSTRUCTIONS.md` — until an arXiv v2 is submitted, the
paper of record does not link to this repository.

## Licensing at a glance

The dataset is composite-licensed and **inherits restrictions**: MaScQA is
CC-BY-NC-SA-4.0 (NonCommercial) and LAB-Bench is CC-BY-SA-4.0 with an upstream
do-not-train request. Omni-MATH-2 is Apache-2.0; JEEBench and SciBench are MIT.
See the dataset's `LICENSE`, `NOTICE` and `LICENSES/README.md`.

## Citation

See `CITATION.cff` in this repository, or the Citation section of the website.
