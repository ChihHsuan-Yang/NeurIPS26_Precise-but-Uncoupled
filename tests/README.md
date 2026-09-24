# tests/

```bash
make test        # or: python3 -m pytest tests/ -q
```

**43 tests in two groups.**

## Pipeline tests (25)

`test_aggregate_release_transitions.py` and `test_symmetric_transition_extractor.py`
are the original unit tests for the 2×5 transition pipeline, recovered verbatim from
the post-rebuttal analysis. The only edit is the path to the module under test; every
assertion is unchanged.

## Release-integrity guards (18)

`test_release_integrity.py` guards defects that were **actually present** in the
source material and fixed during packaging, so they cannot silently return:

| Guard | The defect it prevents |
|---|---|
| no private absolute paths | four scripts shipped with hardcoded author paths |
| no credentials | — |
| no resolvable private endpoint | the client hardcoded a facility hostname |
| single analysis definition | the same 21 modules existed twice, and had diverged |
| entry points import | the extracted analysis tree could not run standalone |
| no import side effects | importing the code wrote log files into the source tree |
| pinned digests | the two load-bearing checksums, checked directly |
| corpus is N=4,181 over ten tiers | a changed corpus would silently change every table |
| retired claim absent | a withdrawn claim must never reappear |
| no withdrawn cost values | the invalidated token/cost numbers must not be quoted |

### These guards are tested against failure

A scanner that matches nothing passes on everything. So:

* every pattern has a **positive control** — it must match a known-bad string, or the
  test fails;
* the walker asserts it reaches a plausible file count;
* the only two files exempt from scanning are the two scanner files themselves (they
  necessarily contain the forbidden patterns as controls), and a test asserts the
  exemption is **exactly those two files** — a new file in `scripts/` or `tests/` is
  still scanned.

During development the suite was run against a tree with six deliberately planted
defects — a private path buried in a vendored module, the retired claim, a withdrawn
cost value, a stray `.DS_Store`, a corrupted pinned digest, and a duplicate analysis
tree — and caught all six.

`scripts/validate_release.py` runs the same checks more strictly (it also fails on
`__pycache__` and log files, which running pytest itself creates) and is meant for a
clean tree.
