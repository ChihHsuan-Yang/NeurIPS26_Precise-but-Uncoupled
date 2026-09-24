#!/usr/bin/env bash
# Download the released dataset from the PUBLIC Hugging Face repo.
#
# Target: https://huggingface.co/datasets/AgentsSci/NeurIPS26_Precise-but-Uncoupled
#
# This release is SELF-CONTAINED. It does NOT read the authors' internal upstream
# dataset (AgentsSci/scientific-agent-protocol-traces), which is private and
# returns HTTP 401 to anonymous users. If you find a reference to that repo
# anywhere, it is provenance information, not a download instruction.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

DEST="${1:-$PU_RELEASE_ROOT}"

rule
say "fetch-data"
say "  dataset : $HF_DATASET_ID"
say "  dest    : $DEST"
rule

if ! "$PYTHON" -c 'import huggingface_hub' >/dev/null 2>&1; then
  die "huggingface_hub is not installed. Run:  pip install -r requirements.txt"
fi

mkdir -p "$DEST"

set +e
"$PYTHON" - "$HF_DATASET_ID" "$DEST" <<'PYEOF'
import sys
from huggingface_hub import snapshot_download

repo_id, dest = sys.argv[1], sys.argv[2]
try:
    path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=dest)
except Exception as exc:  # noqa: BLE001 - we want the message, whatever it is
    msg = str(exc)
    print("DOWNLOAD FAILED: %s" % msg, file=sys.stderr)
    if "401" in msg or "403" in msg or "gated" in msg.lower() or "authentication" in msg.lower():
        print("", file=sys.stderr)
        print("A 401/403 here is EXPECTED BEFORE PUBLICATION.", file=sys.stderr)
        print("The dataset is built private and is flipped public at the final", file=sys.stderr)
        print("release step. If you are reading the published paper and still see", file=sys.stderr)
        print("this, the flip has not happened yet -- please open a GitHub issue.", file=sys.stderr)
        print("If you are an author with access, run `huggingface-cli login` first.", file=sys.stderr)
    sys.exit(1)
print("downloaded to: %s" % path)
PYEOF
rc=$?
set -e

if [ "$rc" -ne 0 ]; then
  rule
  say "fetch-data: FAILED (see message above)"
  say ""
  say "Track A can still run WITHOUT this download: the repository ships the"
  say "pinned 2x5 reference table and the paper's derived report CSVs under"
  say "results/derived_tables/. What needs the download is regenerating the"
  say "transition intermediate from raw traces. See docs/DATA.md."
  exit "$rc"
fi

rule
say "fetch-data: OK"
say "Next:  make verify-checksums"
