# Troubleshooting

---

## `make reproduce-analysis` prints FAIL

The regenerated table does not match the released digest. In order of likelihood:

1. **The input was modified or truncated.** Run `make verify-checksums`. The script
   also prints the input digest before using it; it should be
   `7441a770e590da04cb0accbb381d9a012e905e93b1d7120f2c04b1b53e21023c`.
2. **You pointed it at a regenerated input.** If you set `PU_TRANSITIONS` to a CSV you
   rebuilt from raw traces, the *input* digest will differ (documented and expected) —
   but the *table* should still match. If the table also differs, diff it against
   `results/derived_tables/matrix_2x5_precise_uncoupled_strict.csv`; the script prints
   the first 40 differing lines for you.
3. **A pandas version that changes CSV formatting.** Unlikely — the digest is stable
   across pandas 2.3.3 and 3.0.3 — but pin `pandas==2.3.3` to rule it out.

## `ModuleNotFoundError: No module named 'agentverse'`

Run from the repository root, or set `PYTHONPATH=src`. The `scripts/` wrappers do this
for you; direct `python3 src/...` invocations do not.

## `ImportError: cannot import name 'open' from 'builtins'`

You have put `src/precise_uncoupled` on `PYTHONPATH`. Don't — it contains subpackages
named `io` and `scripts` which shadow the standard library. **Only `src` goes on the
path.** (The Track A process scripts import each other by bare module name; that works
because Python puts a script's own directory on `sys.path[0]` automatically.)

## `ModuleNotFoundError: No module named 'colorama'` / `pydantic` / `tiktoken`

`pip install -r requirements.txt`. Track A needs `pandas`, `pyarrow`, `PyYAML`,
`colorama`; Track B adds `openai`, `pydantic` (v1 line), `tenacity`, `aiohttp`,
`tiktoken`, `rapidfuzz`.

## `TypeError: unsupported operand type(s) for |` on import

Python older than 3.10 hitting a PEP 604 annotation in a module without
`from __future__ import annotations`. Use Python **3.9.6** (verified working — the
affected module is only reached on the Track B path) or **3.13**. Track A is fine on
3.9.

## `make smoke-live` exits 2 without calling anything

`OPENAI_BASE_URL` or `OPENAI_API_KEY` is unset. That is the gate working. Export both
and rerun. The script never contacts any endpoint but yours.

## `make smoke-live` fails with connection errors

Check the endpoint is reachable and OpenAI-compatible:

```bash
curl -s -H "Authorization: Bearer $OPENAI_API_KEY" "$OPENAI_BASE_URL/models" | head
```

The client retries with backoff on 429 and transient 5xx, then raises.

## `Dataset path does not exist: .../configs/paper/per_hint_llm/data/...`

A **relative** `--dataset_path` is resolved against the *config* directory, not your
working directory. Pass an absolute path:
`--dataset_path "$PWD/data/example/omni_math_example_3.jsonl"`. The shipped scripts
already do.

## `make fetch-data` returns HTTP 401

Expected before publication — the dataset is flipped public at the final release step.
If the paper is published and this persists, open an issue. Track A still works
without the download: the pinned intermediate and all reference tables ship in-repo.

## Log files or `__pycache__` appear after I run things

Normal and gitignored. The vendored runtime writes `logs/activity.log` in your working
directory; set `AGENTVERSE_LOG_DIR` to move it. It no longer writes inside the package.
`make clean` removes regenerated outputs and caches.

## `make` fails with a syntax error

Check `make -v`. The Makefile targets GNU Make 3.81 (macOS default) and avoids
`.ONESHELL`, `!=` and grouped targets. If your `make` is stranger than that, run the
`scripts/*.sh` equivalents directly — every target is one script, and Make is never
required.

## Numbers from a live run don't match the paper

They are not supposed to. There is no seed, and the protocol code post-dates the
submitted run. Expected envelope: PER FinalPass ±~0.5 pp, Broadcast ±~2 pp between
independent executions. Compare directions and separations, not absolute percentages.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) §2.3.

## I want to check the whole tree before trusting it

```bash
make validate     # structural + anonymity + digest self-check
make test         # 43 tests: 25 pipeline + 18 release guards
```

Both carry positive controls, so a broken check fails loudly rather than reporting a
clean result.
