# Shared helpers. Sourced by every script in this directory.
# POSIX-ish bash; no GNU-only tools (this must work on macOS as shipped).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
# ONLY src/ goes on the path. Do NOT add src/precise_uncoupled: it contains
# subpackages named `io` and `scripts`, which would SHADOW the standard library
# `io` module and the repo's own scripts/ directory. (This bit us once: adding it
# made every python3.9 subprocess die with
# "ImportError: cannot import name 'open' from 'builtins'".)
# The process/*.py scripts import each other by bare module name; that works
# because Python puts a script's OWN directory on sys.path[0] when you run it.
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python3}"

# Where the downloaded Hugging Face dataset lives.
PU_RELEASE_ROOT="${PU_RELEASE_ROOT:-$REPO_ROOT/data/hf/NeurIPS26_Precise-but-Uncoupled}"
export PU_RELEASE_ROOT

HF_DATASET_ID="AgentsSci/NeurIPS26_Precise-but-Uncoupled"
export HF_DATASET_ID

# sha256 of one file, printed bare. macOS has shasum; Linux usually has sha256sum.
sha256_of() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    "$PYTHON" -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"
  fi
}

say()  { printf '%s\n' "$*"; }
rule() { printf '%s\n' "------------------------------------------------------------------"; }

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
