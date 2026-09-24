"""Trace parsing and dataset loading -- facade over `agentverse` + `dataloader`.

`parse_trace_file` is the single entry point every Track A metric uses to turn a
saved `.trace.txt` into `TraceExample` objects. `OmniMathLoader` is the loader the
paper's `dataset_loader: omni-math` configs select.

Single definition rule: re-exports only.
"""

from agentverse.metrics.repair import (  # noqa: F401
    FeedbackEpisode,
    ReviewEpisode,
    TraceExample,
    parse_trace_file,
)
from agentverse.metrics.api_accounting import (  # noqa: F401
    write_api_accounting_sidecar,
)
from agentverse.metrics.candidate_label_cache import (  # noqa: F401
    default_candidate_label_cache_path,
)
from dataloader.omni_math import OmniMathLoader  # noqa: F401

__all__ = [
    "FeedbackEpisode",
    "OmniMathLoader",
    "ReviewEpisode",
    "TraceExample",
    "default_candidate_label_cache_path",
    "parse_trace_file",
    "write_api_accounting_sidecar",
]
