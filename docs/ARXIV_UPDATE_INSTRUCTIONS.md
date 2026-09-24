# ARXIV UPDATE INSTRUCTIONS — author action required (the sole remaining external step)

arXiv 2607.15388 is at v1 (16 Jul 2026) and has never been revised. Its Comments field
currently advertises ONLY the HF Space:
  "48 pages, 20 figures, 25 tables. Public release website:
   https://huggingface.co/spaces/AgentsSci/scientific-agent-protocol-traces-site"
There is NO GitHub URL in v1 (verified absent by grep of the arXiv clone source), and the
NeurIPS checklist open-access question is answered \answerNo{} with the justification
"Code and data will be released after institutional approval."

That approval condition is now satisfied by this release. Until you submit v2, the paper of
record does not link to the artifacts, and the release documentation says so rather than
implying mutual cross-linking is complete.

## 1. Replace the availability paragraph
FILE: sections/07_limitations_and_broader_impact.tex  (paragraph "Data and code.")
ALSO: sections/08_appendix.tex, appendix "Reproducibility Artifact, Compute Path, and Asset Licenses"

REPLACEMENT TEXT:
  \paragraph{Data and code.}
  Code, configurations, and analysis scripts are available at
  \url{https://github.com/ChihHsuan-Yang/NeurIPS26_Precise-but-Uncoupled}.
  The released traces, process metrics, and derived tables are available at
  \url{https://huggingface.co/datasets/AgentsSci/NeurIPS26_Precise-but-Uncoupled}.
  The project website is \url{https://chihhsuan-yang.github.io/NeurIPS26_Precise-but-Uncoupled/}.
  The offline analysis path reproduces the paper's process tables without model calls; the
  released dataset inherits NonCommercial terms from MaScQA (CC-BY-NC-SA-4.0) and ShareAlike
  plus an upstream do-not-train request from LAB-Bench (CC-BY-SA-4.0).

## 2. Update the arXiv Comments metadata field to
  48 pages, 20 figures, 25 tables. Code: https://github.com/ChihHsuan-Yang/NeurIPS26_Precise-but-Uncoupled
  Data: https://huggingface.co/datasets/AgentsSci/NeurIPS26_Precise-but-Uncoupled
  Website: https://chihhsuan-yang.github.io/NeurIPS26_Precise-but-Uncoupled/

## 3. Flip the NeurIPS checklist open-access answer
\answerNo{} -> \answerYes{} with the URLs above.

## 4. CORRECTIONS YOU SHOULD CONSIDER MAKING IN THE SAME v2
These are not cosmetic. v1 prints numbers that the rebuttal audit withdrew.
 a. v1 PRINTS THE INVALID TOKEN/CALL VALUES AS FACT. Per P0E (confirmed independently at code
    and data level), legacy absolute per-problem token/call fields are cumulative-within-worker
    and exact recovery is impossible. Affected: Table 1 token columns, Table 5 (PER-inner6
    tokens), Table 14 AvgTokens, Tables 20/21 (verifier-cost thresholds), Figures 14/15/16.
    Withdraw the absolute values and the token-caliper sensitivity; keep wall-clock, which is valid.
 b. Do not introduce the retired uptake-ordering claim (that PER takes up critique less often
    in EVERY setting). It is NOT in v1
    (verified by grep) and the symmetric audit measures 0/10 for uptake and 3/10 for repair.
    If you describe the 2x5 audit, the supported wording is: reviewer detection quality and
    successful repair separate broadly across models and domains.
 c. If you add the 2x5 matrix, label it post-rebuttal ROBUSTNESS evidence over five benchmarks
    and two actor families, distinct from the paper's Omni-MATH-only primary claim, and keep the
    small-denominator cells visible (JEEBench/Gemma-4 has n=2 evaluable follow-ups).

## 5. Do not claim acceptance
Reviews were 4/4/4 with the AC leaning accept. No decision may be asserted anywhere.

## 6. After you submit v2
Re-run the cross-link check; only then is the matrix complete on all four surfaces.
