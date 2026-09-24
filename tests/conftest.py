"""Make both import roots available to the test suite.

  src/                          -> `agentverse`, `agentverse_command`, `precise_uncoupled`
  src/precise_uncoupled/process -> the Track A scripts, which import each other by
                                   BARE module name exactly as they did upstream
                                   (that is how they are invoked as scripts too).

The process directory is appended, never prepended, and only for tests: it must
not shadow anything in the standard library.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
PROCESS = SRC / "precise_uncoupled" / "process"

for p in (str(SRC),):
    if p not in sys.path:
        sys.path.insert(0, p)
if str(PROCESS) not in sys.path:
    sys.path.append(str(PROCESS))
