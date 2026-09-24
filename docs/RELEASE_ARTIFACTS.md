# Release Artifacts — all live links

Every URL below was verified with an anonymous request on 2026-09-24; all
returned HTTP 200. Re-check with `bash scripts/check_links.sh`.

## 1. Primary artifacts

| # | Artifact | URL |
|---|---|---|
| 1 | Paper — arXiv abstract | https://arxiv.org/abs/2607.15388 |
| 2 | Paper — PDF | https://arxiv.org/pdf/2607.15388 |
| 3 | Paper — DOI | https://doi.org/10.48550/arXiv.2607.15388 |
| 4 | Code — GitHub repository | https://github.com/ChihHsuan-Yang/NeurIPS26_Precise-but-Uncoupled |
| 5 | Website — GitHub Pages | https://chihhsuan-yang.github.io/NeurIPS26_Precise-but-Uncoupled/ |
| 6 | Data — Hugging Face dataset | https://huggingface.co/datasets/AgentsSci/NeurIPS26_Precise-but-Uncoupled |
| 7 | Model card | **Not applicable — no project-trained model exists** |

## 2. Entry points inside those artifacts

| # | Resource | URL |
|---|---|---|
| 8 | Release tag v1.0.0 | https://github.com/ChihHsuan-Yang/NeurIPS26_Precise-but-Uncoupled/releases/tag/v1.0.0 |
| 9 | Website source branch | https://github.com/ChihHsuan-Yang/NeurIPS26_Precise-but-Uncoupled/tree/gh-pages |
| 10 | README | https://github.com/ChihHsuan-Yang/NeurIPS26_Precise-but-Uncoupled/blob/main/README.md |
| 11 | This file | https://github.com/ChihHsuan-Yang/NeurIPS26_Precise-but-Uncoupled/blob/main/docs/RELEASE_ARTIFACTS.md |
| 12 | Reproducibility guide | https://github.com/ChihHsuan-Yang/NeurIPS26_Precise-but-Uncoupled/blob/main/docs/REPRODUCIBILITY.md |
| 13 | Paper-to-artifact map | https://github.com/ChihHsuan-Yang/NeurIPS26_Precise-but-Uncoupled/blob/main/docs/PAPER_TO_ARTIFACT_MAP.md |
| 14 | arXiv v2 instructions (author action) | https://github.com/ChihHsuan-Yang/NeurIPS26_Precise-but-Uncoupled/blob/main/docs/ARXIV_UPDATE_INSTRUCTIONS.md |
| 15 | Citation metadata | https://github.com/ChihHsuan-Yang/NeurIPS26_Precise-but-Uncoupled/blob/main/CITATION.cff |
| 16 | Data card | https://huggingface.co/datasets/AgentsSci/NeurIPS26_Precise-but-Uncoupled/blob/main/README.md |
| 17 | Dataset viewer | https://huggingface.co/datasets/AgentsSci/NeurIPS26_Precise-but-Uncoupled/viewer |

## 3. Related surfaces (not this paper's artifacts)

| # | Resource | URL | Note |
|---|---|---|---|
| 18 | HF Papers page | https://huggingface.co/papers/2607.15388 | auto-generated; 0 upvotes, no Collection |
| 19 | AgentsSci Space | https://huggingface.co/spaces/AgentsSci/scientific-agent-protocol-traces-site | broader multi-paper resource. **This is the URL arXiv v1's Comments field advertises**, and it does not link onward to items 4-6 |
| 20 | Author publications page | https://chihhsuan-yang.github.io/publications/ | links to Project / Code / Data |
| 21 | Author projects page | https://chihhsuan-yang.github.io/projects/ | links to Project / Code / Data |

## 4. Pinned revisions

| Component | Revision |
|---|---|
| Code, `main` | `467648589f843265af94a354a6f15c48374d68a6` |
| Website, `gh-pages` | `857b579784b9a26780e26a4d98f88a7abf1edab5` |
| Dataset, `main` | `588dc89e1b12aae041cbe015a37c0081c6bd58d6` |
| Author site, `master` | `26311afc73da42a4568010ddfff5478642d031a2` |
| Paper | arXiv:2607.15388**v1**, 16 July 2026, cs.AI |

## 5. Content digests used for reproduction

| Artifact | sha256 |
|---|---|
| `data/derived/full_release_symmetric_transitions.csv` | `7441a770e590da04cb0accbb381d9a012e905e93b1d7120f2c04b1b53e21023c` |
| 2x5 headline table | `6c77c01e6a02889cbf69b77d3f33d0f8e5784a150d3d1b29d9311b06435daef2` |

`make reproduce-analysis` regenerates the second from the first and fails if the
digest differs. No model endpoint, no network.

## 6. Experiment provenance

| | |
|---|---|
| Submitted experiment base | `be9c47c90c95b44d60e2b3bdad49d68b11557039` |
| Resource-accounting patch | `b4a2db6ee53811b1a578972421da6a9cc2a18286` |

## 7. Link graph — one gap remains

| From | arXiv | GitHub | Website | HF dataset |
|---|---|---|---|---|
| **arXiv** | — | missing | missing | missing |
| **GitHub** | yes | — | yes | yes |
| **Website** | yes | yes | — | yes |
| **HF dataset** | yes | yes | yes | — |

The three release surfaces cross-link completely. arXiv is the only node with no
route into them: its Comments field points at item 19, which does not link
onward. Closing this needs an arXiv v2 — see item 14. **Until then, do not
describe cross-linking as complete.**

## 8. Licensing

Composite and **inherited**: MaScQA is CC-BY-NC-SA-4.0 (NonCommercial) and
LAB-Bench is CC-BY-SA-4.0 with an upstream do-not-train request. Omni-MATH-2 is
Apache-2.0; JEEBench and SciBench are MIT. See the dataset's `LICENSE`,
`NOTICE` and `LICENSES/README.md`.

Note: the dataset card carries `extra_gated_prompt` text, but the API reports
`gated: false` and files download unauthenticated. It is display text, not an
access control — an open author decision, recorded in item 14.
