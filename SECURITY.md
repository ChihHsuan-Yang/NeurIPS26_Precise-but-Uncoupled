# Security and privacy

## Reporting a leaked credential or private information

If you find anything in this repository that should not be public — an API key, a
token, a private hostname or endpoint URL, an internal filesystem path, or personal
information — **please do not open a public issue.**

Email the corresponding author directly: **bellayang@anl.gov**, with "SECURITY" in
the subject. Please include the file path, line number, and what you found. We will
confirm receipt and, if the report is valid, remove the material and note the change.

## What this repository deliberately contains

So that you can tell a finding from a non-finding:

**Present on purpose:**

* `bellayang@anl.gov` — the corresponding author's address, already published on the
  arXiv preprint.
* Coarse public facility and backend labels (for example `aurora`, `crux`, `sophia`)
  where they appear as **data values** in released records. These are public
  supercomputer names, not hostnames, and they are load-bearing: the run-to-run
  variability analysis is stratified by backend, so removing them would make that
  analysis unreproducible.
* Public model identifiers and public dataset URLs.

**Absent on purpose, and checked:**

* API keys, tokens, bearer credentials, cookies, `.env` files, SSH keys.
* Resolvable private endpoint URLs, IP addresses, ports.
* Usernames, and absolute filesystem paths of any kind -- home directories on
  macOS or Linux, and cluster storage paths.
* `.git` directories copied from any source tree.

Both `scripts/validate_release.py` and `tests/test_release_integrity.py` scan for all
of the above. Both carry **positive controls**: if a detection pattern stops matching
its own known-bad example, the check fails rather than silently reporting a clean
tree. Run them yourself:

```bash
make validate
make test
```

## Running this code safely

* **Your credentials stay yours.** `OPENAI_API_KEY` and `OPENAI_BASE_URL` are read
  from the environment and are never written to disk by this code. `make smoke-live`
  prints the key's *length* to confirm it is set, never its value.
* **No endpoint of ours is contacted.** There is no default endpoint. Live runs go
  only where you point them.
* **No telemetry.** Nothing here phones home.
* **Optional network access** occurs in exactly two places, both explicit:
  `make fetch-data` (Hugging Face) and Track B (your endpoint).
* **Untrusted input:** the analysis parses saved trace files. If you run it over
  traces from a source you do not trust, treat that as you would any untrusted input.

## Scope

This is research code accompanying a paper, not a maintained security product. There
is no support commitment and no patch schedule. It is published so results can be
verified.
