import logging
import json
import ast
import asyncio
import os
import shlex
import random
import subprocess
import sys
import time
from collections import deque
import threading
import hashlib
import numpy as np
from aiohttp import ClientSession
from pathlib import Path
from typing import Dict, List, Optional, Union
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from pydantic import Field

from agentverse.llms.base import LLMResult
from agentverse.logging import logger
from agentverse.message import Message

from . import llm_registry, LOCAL_LLMS, LOCAL_LLMS_MAPPING
from .base import BaseChatModel, BaseModelArgs
from .utils.jsonrepair import JsonRepair
from .utils.llm_server_utils import get_llm_server_modelname
from .utils import count_message_tokens

try:
    from openai import OpenAI, AsyncOpenAI
    from openai import OpenAIError
    from openai import AzureOpenAI, AsyncAzureOpenAI
except ImportError:
    is_openai_available = False
    logger.warn(
        "openai package is not installed. Please install it via `pip install openai`"
    )

# Fallback imports for the error classes the rate-limit/transient retry
# wrapper detects. Modern openai>=1.x exports these; older SDKs may not.
# If absent, set to None so isinstance() guards short-circuit safely.
try:
    from openai import RateLimitError as _RateLimitError  # type: ignore
except Exception:  # pragma: no cover
    _RateLimitError = None  # type: ignore[assignment]
try:
    from openai import APIConnectionError as _APIConnectionError  # type: ignore
except Exception:  # pragma: no cover
    _APIConnectionError = None  # type: ignore[assignment]
try:
    from openai import APITimeoutError as _APITimeoutError  # type: ignore
except Exception:  # pragma: no cover
    _APITimeoutError = None  # type: ignore[assignment]
else:
    api_key = None
    base_url = None
    model_name = None
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
    AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
    AZURE_API_BASE = os.environ.get("AZURE_OPENAI_API_BASE")
    VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL")
    VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
    DEFAULT_CLIENT = None
    DEFAULT_CLIENT_ASYNC = None

    if not OPENAI_API_KEY and not AZURE_API_KEY:
        logger.warn(
            "OpenAI API key is not set. Please set an environment variable OPENAI_API_KEY or "
            "AZURE_OPENAI_API_KEY."
        )
    elif OPENAI_API_KEY:
        DEFAULT_CLIENT = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        DEFAULT_CLIENT_ASYNC = AsyncOpenAI(
            api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL
        )
        api_key = OPENAI_API_KEY
        base_url = OPENAI_BASE_URL
    elif AZURE_API_KEY:
        DEFAULT_CLIENT = AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_API_BASE,
            api_version="2024-02-15-preview",
        )
        DEFAULT_CLIENT_ASYNC = AsyncAzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_API_BASE,
        )
        api_key = AZURE_API_KEY
        base_url = AZURE_API_BASE
    if VLLM_BASE_URL:
        if model_name := get_llm_server_modelname(VLLM_BASE_URL, VLLM_API_KEY, logger):
            # model_name = /mnt/llama/hf_models/TheBloke_Llama-2-70B-Chat-GPTQ
            # transform to TheBloke/Llama-2-70B-Chat-GPTQ
            hf_model_name = model_name.split("/")[-1].replace("_", "/")
            LOCAL_LLMS.append(model_name)
            LOCAL_LLMS_MAPPING[model_name] = {
                "hf_model_name": hf_model_name,
                "base_url": VLLM_BASE_URL,
                "api_key": VLLM_API_KEY if VLLM_API_KEY else "EMPTY",
            }
            logger.info(f"Using vLLM model: {hf_model_name}")
    # RELEASE PATCH (NeurIPS26_Precise-but-Uncoupled): upstream AgentVerse probed
    # http://localhost:5000 for an FSChat server on EVERY import. That is a network
    # call at import time, it is not used by any paper config, and on macOS port
    # 5000 is taken by the AirPlay Receiver -- which made the probe print a
    # misleading "vLLM server is running ... Status code: 403" on a clean machine.
    # The probe is now opt-in via FSCHAT_BASE_URL.
    _FSCHAT_BASE_URL = os.environ.get("FSCHAT_BASE_URL")
    if _FSCHAT_BASE_URL and (
        hf_model_name := get_llm_server_modelname(_FSCHAT_BASE_URL, logger=logger)
    ):
        # meta-llama/Llama-2-7b-chat-hf
        # transform to llama-2-7b-chat-hf
        short_model_name = model_name.split("/")[-1].lower()
        LOCAL_LLMS.append(short_model_name)
        LOCAL_LLMS_MAPPING[short_model_name] = {
            "hf_model_name": hf_model_name,
            "base_url": "http://localhost:5000/v1",
            "api_key": "EMPTY",
        }

        logger.info(f"Using FSChat model: {model_name}")

# ----------------------------
# API frequency monitor (calls/sec)
# ----------------------------

# --- Per-user, cross-process rate monitor (file-locked) ---
def _key_fingerprint() -> str:
    # stable fingerprint (do NOT log the raw key)
    k = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("AZURE_OPENAI_API_KEY", "")
    if not k:
        return "nokey"
    return hashlib.sha256(k.encode("utf-8")).hexdigest()[:12]

_KEY_FP = os.environ.get("AGENTVERSE_KEY_FP", _key_fingerprint())

_RATE_STATE_PATH = os.environ.get(
    "AGENTVERSE_RATE_STATE",
    f"/tmp/agentverse_rate_{os.environ.get('USER','unknown')}_{_KEY_FP}.json"
)
_RATE_LOCK_PATH = _RATE_STATE_PATH + ".lock"


# ----------------------------
# Per-user cross-process rate monitor config
# ----------------------------

_RATE_STATE_PATH = os.environ.get(
    "AGENTVERSE_RATE_STATE",
    f"/tmp/agentverse_rate_{os.environ.get('USER','unknown')}.json"
)
_RATE_LOCK_PATH = _RATE_STATE_PATH + ".lock"

_API_WINDOW_SEC = 1.0
_API_RATE_LIMIT = 20

# Headroom against the server-side ~20 req/s ceiling. Below this the worker
# may issue a request immediately; at-or-above it sleeps in the pre-call gate
# until the 1-second window slides. Overridable via env so we can tighten
# (e.g. AGENTVERSE_RATE_SOFT_LIMIT=8) when the endpoint is degraded without
# touching code.
_PRE_CALL_THROTTLE_SOFT_LIMIT = int(os.environ.get("AGENTVERSE_RATE_SOFT_LIMIT", "18"))

# Backoff caps and attempt budgets for the two recoverable error classes.
# Rate-limit gets a longer budget because 429s typically self-clear in <60s;
# transient (connection / timeout) errors get a shorter budget because the
# agent layer also retries.
_RATE_LIMIT_MAX_ATTEMPTS    = int(os.environ.get("AGENTVERSE_RATE_LIMIT_MAX_ATTEMPTS", "7"))
_RATE_LIMIT_BACKOFF_CAP_SEC = float(os.environ.get("AGENTVERSE_RATE_LIMIT_BACKOFF_CAP", "60"))
_TRANSIENT_MAX_ATTEMPTS     = int(os.environ.get("AGENTVERSE_TRANSIENT_MAX_ATTEMPTS", "4"))
_TRANSIENT_BACKOFF_CAP_SEC  = float(os.environ.get("AGENTVERSE_TRANSIENT_BACKOFF_CAP", "8"))
_GATE_SINGLE_SLEEP_CAP_SEC  = 60.0


# RELEASE PATCH (NeurIPS26_Precise-but-Uncoupled).
# Upstream these two helpers hardcoded the authors' facility endpoint hostname and
# one of its URL path segments, to work around two deployment quirks:
#   (a) that deployment capped some Llama-3.1 endpoints at a 16k window;
#   (b) one of its gateways required a shortened model alias.
# A resolvable endpoint URL must not ship in a public release, and neither quirk
# applies to a reader's own endpoint. Both are now OPT-IN and generic:
#
#   AGENTVERSE_ENDPOINT_QUIRKS_HOST   substring of your base_url that marks an
#                                     endpoint needing the workarounds (unset = off)
#   AGENTVERSE_MODEL_ALIASES          "from=to,from=to" model-id rewrites applied
#                                     when that host matches (unset = no rewriting)
#
# Left unset -- the default for every reader -- both functions are inert: the
# endpoint check is always False and model ids pass through untouched.
_ENDPOINT_QUIRKS_HOST = os.environ.get("AGENTVERSE_ENDPOINT_QUIRKS_HOST", "")


def _is_quirked_endpoint(base_url: str | None) -> bool:
    if not _ENDPOINT_QUIRKS_HOST:
        return False
    return _ENDPOINT_QUIRKS_HOST in str(base_url or "")


def _model_alias_map() -> dict:
    raw = os.environ.get("AGENTVERSE_MODEL_ALIASES", "")
    aliases = {}
    for pair in raw.split(","):
        if "=" in pair:
            src, _, dst = pair.partition("=")
            src, dst = src.strip(), dst.strip()
            if src and dst:
                aliases[src] = dst
    return aliases


def _normalize_endpoint_model_name(model: str | None, *, base_url: str | None) -> str:
    raw_model = str(model or "")
    if not _is_quirked_endpoint(base_url):
        return raw_model
    return _model_alias_map().get(raw_model, raw_model)


def _is_auth_error(error: Exception, *, base_url: str | None) -> bool:
    """True for errors that look like an expired/invalid bearer token."""
    raw = str(error or "").lower()
    markers = (
        "token introspection",
        "token is either not active or invalid",
        "authenticationerror",
        "error code: 401",
        "status code: 401",
        "permission denied from internal policies",
        "high-assurance timeout",
    )
    return any(marker in raw for marker in markers)


def _refresh_access_token(client_args: Dict, logger_obj=None) -> bool:
    """Optionally re-mint a short-lived API token after a 401.

    RELEASE PATCH (NeurIPS26_Precise-but-Uncoupled): upstream this shelled out to
    `inference_auth_token.py`, a Globus/facility-specific helper that is NOT part
    of this release. It is now a generic opt-in hook:

        AGENTVERSE_TOKEN_REFRESH_CMD="/path/to/your/mint-token.sh"

    The command must print the new bearer token on stdout. Unset (the default)
    means no refresh is attempted and a 401 is raised to the caller, which is the
    right behaviour for a static API key.
    """
    refresh_cmd = os.environ.get("AGENTVERSE_TOKEN_REFRESH_CMD", "").strip()
    if not refresh_cmd:
        return False

    completed = subprocess.run(
        shlex.split(refresh_cmd),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        if logger_obj is not None:
            logger_obj.warning(
                "[TOKEN REFRESH] failed: returncode=%s stderr=%s",
                completed.returncode,
                (completed.stderr or "").strip()[:1000],
            )
        return False

    token = (completed.stdout or "").strip().splitlines()
    token = token[-1].strip() if token else ""
    if not token:
        if logger_obj is not None:
            logger_obj.warning("[TOKEN REFRESH] command returned an empty access token.")
        return False

    os.environ["OPENAI_API_KEY"] = token
    globals()["OPENAI_API_KEY"] = token
    client_args["api_key"] = token

    if logger_obj is not None:
        logger_obj.info("[TOKEN REFRESH] Refreshed OpenAI-compatible endpoint access token.")
    return True


def _acquire_lock(lock_path: str, timeout: float = 2.0, poll: float = 0.01):
    """
    Simple cross-process lock using atomic file create.
    Prevents multiple processes from writing the JSON file simultaneously.
    """
    start = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            if time.time() - start > timeout:
                # Best effort: don't hang forever
                return
            time.sleep(poll)


def _release_lock(lock_path: str):
    try:
        os.unlink(lock_path)
    except FileNotFoundError:
        pass


def _collapse_consecutive_same_role(messages: List[dict]) -> List[dict]:
    """Normalize messages for strict-template models (gemma3, llama-instruct).

    vLLM's gemma3 enforces strict `user/assistant/user/assistant/...`
    alternation with at most one leading `system` message. Two failure
    modes from the upstream agent code trip this:

    1. Two consecutive same-role messages (e.g. agent loop appends
       `user` while history already ends with `user`)
    2. `system` messages injected mid-conversation (e.g. memory summary
       in chat_history.py inserts `{role: system, content: summary}`
       between user/assistant turns)

    Both come back from the server as `400 BadRequestError: Conversation
    roles must alternate ...`, which some gateways wrap as HTTP 500 via
    openinference_svc. The retry wrapper doesn't classify those as
    recoverable, so the call burns its full retry budget and returns
    empty.

    Normalization steps:
      a. Pull all `system` messages out; merge them (joined by "\\n\\n")
         into a single leading `system` message. This eliminates
         system-mid-conversation entirely.
      b. Walk remaining user/assistant messages and collapse runs of
         the same role by concatenating content.
      c. If the first non-system message is `assistant`, prepend an
         empty `user` so the conversation starts with `user` (some
         strict templates require it).
      d. List/multimodal content is preserved as a separate entry
         (rare in this codebase; merge falls back to keep-separate).
    """
    if not messages:
        return messages

    # (a) extract system messages
    system_parts: List[str] = []
    rest: List[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            c = msg.get("content", "")
            if isinstance(c, str) and c.strip():
                system_parts.append(c)
            elif isinstance(c, list):
                # Preserve multimodal system content as its own entry
                # (rare; will keep position via pass-through).
                rest.append(dict(msg))
        else:
            rest.append(msg)

    # (b) collapse consecutive same-role in the user/assistant tail
    collapsed: List[dict] = []
    for msg in rest:
        role = msg.get("role")
        content = msg.get("content", "")
        if collapsed and collapsed[-1].get("role") == role:
            prev = collapsed[-1]
            prev_content = prev.get("content", "")
            if isinstance(prev_content, str) and isinstance(content, str):
                prev["content"] = (prev_content + "\n\n" + content) if prev_content else content
                continue
        collapsed.append(dict(msg))

    # (c) ensure conversation starts with user (after any system)
    if collapsed and collapsed[0].get("role") == "assistant":
        collapsed.insert(0, {"role": "user", "content": ""})

    # (d) build final: leading merged system + alternating remainder
    out: List[dict] = []
    if system_parts:
        out.append({"role": "system", "content": "\n\n".join(system_parts)})
    out.extend(collapsed)
    return out


def _choose_base_url(model: str, default_base_url: str) -> str:
    """Per-call base URL routing for split-endpoint deployments.

    When OPENAI_BASE_URL_GEMMA_DEV or OPENAI_BASE_URL_GEMMA3_DEV is set and
    `model` is a gemma family model, route ~half the workers to the dev URL based on a hash of
    the worker's PID. All other models (and all workers when the dev URL
    is unset) use `default_base_url`.

    Used by the authors to spread gemma-3 traffic across two backends of the
    same facility when one served the actor model but not the evaluator.
    Readers with a single endpoint can ignore this entirely: leaving the dev-URL
    variables unset makes the function return `default_base_url`. Per-worker
    determinism
    (same PID → same endpoint for the worker's lifetime) keeps vLLM
    prefix-cache hit rates high.

    NOTE: raw `pid % 2` parity is biased on many Linux kernels — they
    allocate PIDs in patterns that skew toward odd or even (we observed
    ~87% odd in one job). MD5(pid) is well-distributed and gives a
    near-50/50 split.
    """
    dev_url = (
        os.environ.get("OPENAI_BASE_URL_GEMMA_DEV")
        or os.environ.get("OPENAI_BASE_URL_GEMMA3_DEV")
        or ""
    ).strip()
    if not dev_url:
        return default_base_url
    if "gemma" not in (model or "").lower():
        return default_base_url
    pid_hash = int(hashlib.md5(str(os.getpid()).encode()).hexdigest(), 16)
    return dev_url if (pid_hash % 2 == 1) else default_base_url


def _read_shared_state():
    """Read the cross-process state file. Caller must hold the lock.

    Returns (times, cooldown_until). Backward compatible with state files
    written before cooldown_until existed (defaults to 0.0).
    """
    try:
        with open(_RATE_STATE_PATH, "r") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            state = {}
        times = state.get("times", [])
        if not isinstance(times, list):
            times = []
        try:
            cooldown_until = float(state.get("cooldown_until", 0.0) or 0.0)
        except (TypeError, ValueError):
            cooldown_until = 0.0
        return times, cooldown_until
    except FileNotFoundError:
        return [], 0.0
    except Exception:
        return [], 0.0


def _write_shared_state(times, cooldown_until: float) -> None:
    """Caller must hold the lock."""
    try:
        with open(_RATE_STATE_PATH, "w") as f:
            json.dump(
                {
                    "times": times,
                    "cooldown_until": cooldown_until,
                    "updated": time.time(),
                    "pid": os.getpid(),
                },
                f,
            )
    except Exception:
        pass


def api_monitor_tick(tag: str = "", logger_obj=None):
    """
    Cross-process per-user requests/sec monitor.

    - Stores recent timestamps in a shared JSON file in /tmp
    - Uses a lock file to prevent races between processes
    - Keeps only timestamps within the last `_API_WINDOW_SEC`
    - Logs and prints the current calls/sec
    - Never raises exceptions (monitor must not crash pipeline)
    """
    now = time.time()
    pid = os.getpid()

    calls_last_window = 0

    _acquire_lock(_RATE_LOCK_PATH)
    try:
        # --------------------
        # Load existing state (preserves cooldown_until set by the
        # rate-limit retry handler so this tick doesn't clobber it)
        # --------------------
        times, cooldown_until = _read_shared_state()

        # --------------------
        # Update timestamps
        # --------------------
        times.append(now)
        cutoff = now - _API_WINDOW_SEC
        times = [t for t in times if isinstance(t, (int, float)) and t >= cutoff]

        calls_last_window = len(times)

        # --------------------
        # Persist back to disk (preserves cooldown_until from the retry handler)
        # --------------------
        _write_shared_state(times, cooldown_until)

    finally:
        _release_lock(_RATE_LOCK_PATH)

    # --------------------
    # Logging / printing
    # --------------------
    msg = (
        f"[API MONITOR][pid={pid}]"
        f"{'[' + tag + ']' if tag else ''} "
        f"calls_last_1s={calls_last_window}"
    )

    if calls_last_window > _API_RATE_LIMIT:
        msg += f"  >>> ABOVE {_API_RATE_LIMIT}/s !!!"

    # Always print to terminal
    print(msg, flush=True)

    # Optional logger
    if logger_obj is not None:
        try:
            if calls_last_window > _API_RATE_LIMIT:
                logger_obj.warning(msg)
            else:
                logger_obj.info(msg)
        except Exception:
            pass

    return calls_last_window
# ----------------------------
# API rate monitor (per-user, cross-process)
_MONITOR_LOG = logging.getLogger("api_monitor")
_MONITOR_LOG.setLevel(logging.INFO)
# RELEASE PATCH (NeurIPS26_Precise-but-Uncoupled): upstream this unconditionally
# opened "api_rate_monitor.log" in the CURRENT WORKING DIRECTORY at import time,
# dropping a log file into whatever directory you happened to run from (including
# a clean checkout). It is now opt-in and defaults to a gitignored path.
#   AGENTVERSE_RATE_LOG=<path>   enable, writing there
#   AGENTVERSE_RATE_LOG=1        enable, writing ./logs/api_rate_monitor.log
# Unset (the default) = no file handler, no file created.
_RATE_LOG = os.environ.get("AGENTVERSE_RATE_LOG", "")
if _RATE_LOG and not _MONITOR_LOG.handlers:
    _rate_log_path = (
        os.path.join("logs", "api_rate_monitor.log") if _RATE_LOG == "1" else _RATE_LOG
    )
    try:
        _parent = os.path.dirname(os.path.abspath(_rate_log_path))
        if _parent:
            os.makedirs(_parent, exist_ok=True)
        fh = logging.FileHandler(_rate_log_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh.setFormatter(formatter)
        _MONITOR_LOG.addHandler(fh)
    except OSError:
        # A log we cannot open must never break a run.
        pass


# ----------------------------------------------------------------------
# Pre-call throttle gate + 429-aware retry/backoff with cross-process
# cooldown coordination.
#
# When the endpoint returns HTTP 429 (rate-limit exceeded), sleep
# before retrying. Honor `Retry-After` header if the server provides it;
# otherwise fall back to exponential backoff with full jitter. The
# cooldown is persisted in the shared `/tmp/agentverse_rate_*.json` file
# so all worker processes pause together instead of stampeding.
#
# Other recoverable errors (`APIConnectionError`, `APITimeoutError`)
# get a smaller, separate retry budget. All other `OpenAIError`
# subclasses re-raise immediately, so the existing auth-retry loop
# (and the agent-side `max_retry`) still catch what they used to.
# ----------------------------------------------------------------------

def _compute_pre_call_sleep() -> float:
    """Returns seconds to sleep before issuing the next call. 0 means go."""
    now = time.time()
    _acquire_lock(_RATE_LOCK_PATH)
    try:
        times, cooldown_until = _read_shared_state()
        if cooldown_until > now:
            return cooldown_until - now
        cutoff = now - _API_WINDOW_SEC
        recent = [t for t in times if isinstance(t, (int, float)) and t >= cutoff]
        if len(recent) >= _PRE_CALL_THROTTLE_SOFT_LIMIT:
            # Sleep until the oldest in-window call rolls off, plus a
            # small skew so we don't all wake up on the same millisecond.
            return max(0.0, (recent[0] + _API_WINDOW_SEC) - now + 0.01)
    finally:
        _release_lock(_RATE_LOCK_PATH)
    return 0.0


def _pre_call_gate_sync(tag: str = "") -> None:
    while True:
        sleep_for = _compute_pre_call_sleep()
        if sleep_for <= 0:
            return
        time.sleep(min(sleep_for, _GATE_SINGLE_SLEEP_CAP_SEC))


async def _pre_call_gate_async(tag: str = "") -> None:
    while True:
        sleep_for = _compute_pre_call_sleep()
        if sleep_for <= 0:
            return
        await asyncio.sleep(min(sleep_for, _GATE_SINGLE_SLEEP_CAP_SEC))


def _register_cooldown(seconds: float) -> None:
    """Bump shared cooldown_until = max(existing, now + seconds)."""
    if seconds <= 0:
        return
    target = time.time() + seconds
    _acquire_lock(_RATE_LOCK_PATH)
    try:
        times, current_cooldown = _read_shared_state()
        new_cooldown = max(current_cooldown, target)
        _write_shared_state(times, new_cooldown)
    finally:
        _release_lock(_RATE_LOCK_PATH)


def _retry_after_seconds_from_error(err) -> Optional[float]:
    """Extract Retry-After (seconds) from an exception's response, if any."""
    response = getattr(err, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = None
    try:
        raw = headers.get("Retry-After") or headers.get("retry-after")
    except Exception:
        raw = None
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _compute_backoff(err, attempt: int, *, cap: float) -> float:
    """Backoff for a recoverable error. Honors Retry-After."""
    retry_after = _retry_after_seconds_from_error(err)
    if retry_after is not None:
        return min(cap, max(0.0, retry_after) + random.uniform(0, 0.5))
    # Exponential backoff with full jitter, attempts 0..N
    return min(cap, (2 ** attempt) + random.uniform(0, 1))


def _is_rate_limit_error(err) -> bool:
    if _RateLimitError is not None and isinstance(err, _RateLimitError):
        return True
    # Defensive: also treat any OpenAIError with status_code 429 as rate-limit
    status = getattr(err, "status_code", None) or getattr(getattr(err, "response", None), "status_code", None)
    return status == 429


def _is_transient_error(err) -> bool:
    if _APIConnectionError is not None and isinstance(err, _APIConnectionError):
        return True
    if _APITimeoutError is not None and isinstance(err, _APITimeoutError):
        return True
    # Treat 5xx as transient — common when a gateway wraps
    # an upstream vLLM error (e.g. 400 BadRequest gets wrapped as 500).
    # Without this the wrapper re-raises immediately on a server hiccup
    # and the agent's outer retry loop hammers the same broken request.
    # Bounded by _TRANSIENT_MAX_ATTEMPTS so we don't loop forever on a
    # genuinely persistent server bug.
    status = (
        getattr(err, "status_code", None)
        or getattr(getattr(err, "response", None), "status_code", None)
    )
    if isinstance(status, int) and 500 <= status < 600:
        return True
    return False


def _log_recoverable(kind: str, attempt: int, max_attempts: int, wait: float, tag: str) -> None:
    msg = (
        f"[LLM] Backing off {wait:.1f}s due to {kind} "
        f"(attempt {attempt + 1}/{max_attempts}). tag={tag}"
    )
    print(msg, flush=True)
    try:
        _MONITOR_LOG.warning(msg)
    except Exception:
        pass


def _client_needs_bare_model_id(client) -> bool:
    """True if this endpoint rejects a provider-prefixed model id.

    RELEASE PATCH (NeurIPS26_Precise-but-Uncoupled): upstream this matched a
    specific URL path segment of the authors' facility gateway, one of whose two
    backends accepted only the bare id `gpt-oss-120b` while the other also
    accepted `openai/gpt-oss-120b`. That is a property of one deployment, not of
    the protocol, so it is now opt-in and generic:

        AGENTVERSE_BARE_MODEL_ID=1        always send the bare id
        AGENTVERSE_BARE_MODEL_ID=<substr> send it when base_url contains <substr>

    Unset (the default) means model ids are passed through exactly as the config
    writes them, which is what a standard OpenAI-compatible server expects.
    """
    marker = os.environ.get("AGENTVERSE_BARE_MODEL_ID", "").strip()
    if not marker:
        return False
    if marker == "1":
        return True
    try:
        return marker in str(getattr(client, "base_url", "") or "")
    except Exception:
        return False


def _maybe_rewrite_model_for_endpoint(client, kwargs):
    """Strip a provider prefix ('openai/') when the endpoint demands a bare id."""
    if not _client_needs_bare_model_id(client):
        return kwargs
    model = kwargs.get("model")
    if not (isinstance(model, str) and model.startswith("openai/")):
        return kwargs
    new_kwargs = dict(kwargs)
    new_kwargs["model"] = model[len("openai/"):]
    return new_kwargs


def call_chat_completion_with_retry(client, *, tag: str, **request_kwargs):
    """Sync wrapper around client.chat.completions.create with rate-limit + transient retries."""
    request_kwargs = _maybe_rewrite_model_for_endpoint(client, request_kwargs)
    last_err = None
    for attempt in range(_RATE_LIMIT_MAX_ATTEMPTS):
        _pre_call_gate_sync(tag=tag)
        api_monitor_tick(tag=tag, logger_obj=_MONITOR_LOG)
        try:
            return client.chat.completions.create(**request_kwargs)
        except Exception as e:
            if _is_rate_limit_error(e):
                wait = _compute_backoff(e, attempt, cap=_RATE_LIMIT_BACKOFF_CAP_SEC)
                _register_cooldown(wait)
                _log_recoverable("RateLimitError", attempt, _RATE_LIMIT_MAX_ATTEMPTS, wait, tag)
                time.sleep(wait)
                last_err = e
                continue
            if _is_transient_error(e):
                if attempt >= _TRANSIENT_MAX_ATTEMPTS - 1:
                    raise
                wait = _compute_backoff(e, attempt, cap=_TRANSIENT_BACKOFF_CAP_SEC)
                _log_recoverable(type(e).__name__, attempt, _TRANSIENT_MAX_ATTEMPTS, wait, tag)
                time.sleep(wait)
                last_err = e
                continue
            raise
    if last_err is not None:
        raise last_err
    raise RuntimeError("retry loop terminated without producing a response (unreachable)")


async def acall_chat_completion_with_retry(client, *, tag: str, **request_kwargs):
    """Async wrapper around client.chat.completions.create with rate-limit + transient retries."""
    request_kwargs = _maybe_rewrite_model_for_endpoint(client, request_kwargs)
    last_err = None
    for attempt in range(_RATE_LIMIT_MAX_ATTEMPTS):
        await _pre_call_gate_async(tag=tag)
        api_monitor_tick(tag=tag, logger_obj=_MONITOR_LOG)
        try:
            return await client.chat.completions.create(**request_kwargs)
        except Exception as e:
            if _is_rate_limit_error(e):
                wait = _compute_backoff(e, attempt, cap=_RATE_LIMIT_BACKOFF_CAP_SEC)
                _register_cooldown(wait)
                _log_recoverable("RateLimitError", attempt, _RATE_LIMIT_MAX_ATTEMPTS, wait, tag)
                await asyncio.sleep(wait)
                last_err = e
                continue
            if _is_transient_error(e):
                if attempt >= _TRANSIENT_MAX_ATTEMPTS - 1:
                    raise
                wait = _compute_backoff(e, attempt, cap=_TRANSIENT_BACKOFF_CAP_SEC)
                _log_recoverable(type(e).__name__, attempt, _TRANSIENT_MAX_ATTEMPTS, wait, tag)
                await asyncio.sleep(wait)
                last_err = e
                continue
            raise
    if last_err is not None:
        raise last_err
    raise RuntimeError("retry loop terminated without producing a response (unreachable)")


def _get_content_or_empty(response, tag: str = "", logger_obj=None) -> str:
    """
    Safely extract message.content from OpenAI-compatible response.
    If content is None/empty, fall back to compatible reasoning fields when present.
    Return "" (never None) so LLMResult(content=...) never triggers
    pydantic/len(None) crashes upstream.
    """
    def _coerce_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = []
            for item in value:
                part = _coerce_text(item)
                if part:
                    parts.append(part)
            return "\n".join(part for part in parts if part).strip()
        if isinstance(value, dict):
            for key in (
                "text",
                "content",
                "value",
                "output_text",
                "reasoning_content",
            ):
                part = _coerce_text(value.get(key))
                if part:
                    return part
            return ""
        return str(value)

    try:
        msg_obj = response.choices[0].message
    except Exception as e:
        # If response is malformed, log and return empty string
        warn_msg = f"[LLM] Malformed response (no choices/message). tag={tag} err={e}"
        if logger_obj is not None:
            logger_obj.warning(warn_msg)
        else:
            print(warn_msg, flush=True)
        return ""

    try:
        debug_payload = msg_obj.model_dump()
    except Exception:
        debug_payload = {}

    content = _coerce_text(getattr(msg_obj, "content", None)).strip()
    if content:
        return content

    fallback_candidates = [
        ("reasoning_content", getattr(msg_obj, "reasoning_content", None)),
        ("output_text", getattr(msg_obj, "output_text", None)),
    ]
    if isinstance(debug_payload, dict):
        fallback_candidates.extend(
            [
                ("reasoning_content", debug_payload.get("reasoning_content")),
                ("output_text", debug_payload.get("output_text")),
                ("reasoning", debug_payload.get("reasoning")),
            ]
        )

    for field_name, raw_value in fallback_candidates:
        fallback_text = _coerce_text(raw_value).strip()
        if fallback_text:
            info_msg = (
                f"[LLM] Using fallback text from {field_name}. tag={tag} "
                f"because message.content was empty."
            )
            if logger_obj is not None:
                logger_obj.info(info_msg)
            else:
                print(info_msg, flush=True)
            return fallback_text

    try:
        finish_reason = getattr(response.choices[0], "finish_reason", None)
    except Exception:
        finish_reason = None

    if not debug_payload:
        try:
            debug_payload = str(msg_obj)
        except Exception:
            debug_payload = "<unavailable>"

    warn_msg = (
        f"[LLM] Empty content from server. tag={tag} finish_reason={finish_reason}. "
        f"message={debug_payload}"
    )
    if logger_obj is not None:
        logger_obj.warning(warn_msg)
    else:
        print(warn_msg, flush=True)

    return ""

def _log_llm_raw(content: str, tag: str, logger_obj=None, also_print: bool = True):
    # show a short preview in terminal
    if also_print:
        preview = (content or "").replace("\n", "\\n")
        print(f"[LLM RAW PREVIEW][{tag}] {preview[:200]}", flush=True)

    # always write the full (truncated) content to file logger
    if logger_obj is not None:
        logger_obj.info(f"[LLM RAW][{tag}] {repr((content or '')[:5000])}")
        # force flush to disk
        for h in logger_obj.handlers:
            try:
                h.flush()
            except Exception:
                pass



class OpenAIChatArgs(BaseModelArgs):
    model: str = Field(default="gpt-3.5-turbo")
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=1.0)
    top_p: int = Field(default=1)
    n: int = Field(default=1)
    stop: Optional[Union[str, List]] = Field(default=None)
    presence_penalty: int = Field(default=0)
    frequency_penalty: int = Field(default=0)


# class OpenAICompletionArgs(OpenAIChatArgs):
#     model: str = Field(default="text-davinci-003")
#     suffix: str = Field(default="")
#     best_of: int = Field(default=1)


# @llm_registry.register("text-davinci-003")
# class OpenAICompletion(BaseCompletionModel):
#     args: OpenAICompletionArgs = Field(default_factory=OpenAICompletionArgs)

#     def __init__(self, max_retry: int = 3, **kwargs):
#         args = OpenAICompletionArgs()
#         args = args.dict()
#         for k, v in args.items():
#             args[k] = kwargs.pop(k, v)
#         if len(kwargs) > 0:
#             logging.warning(f"Unused arguments: {kwargs}")
#         super().__init__(args=args, max_retry=max_retry)

#     def generate_response(self, prompt: str) -> LLMResult:
#         response = openai.Completion.create(prompt=prompt, **self.args.dict())
#         return LLMResult(
#             content=response["choices"][0]["text"],
#             send_tokens=response["usage"]["prompt_tokens"],
#             recv_tokens=response["usage"]["completion_tokens"],
#             total_tokens=response["usage"]["total_tokens"],
#         )

#     async def agenerate_response(self, prompt: str) -> LLMResult:
#         response = await openai.Completion.acreate(prompt=prompt, **self.args.dict())
#         return LLMResult(
#             content=response["choices"][0]["text"],
#             send_tokens=response["usage"]["prompt_tokens"],
#             recv_tokens=response["usage"]["completion_tokens"],
#             total_tokens=response["usage"]["total_tokens"],
#         )


# To support your own local LLMs, register it here and add it into LOCAL_LLMS.
@llm_registry.register("gpt-35-turbo")
@llm_registry.register("gpt-3.5-turbo")
@llm_registry.register("gpt-4")
@llm_registry.register("vllm")
@llm_registry.register("local")
class OpenAIChat(BaseChatModel):
    args: OpenAIChatArgs = Field(default_factory=OpenAIChatArgs)
    client_args: Optional[Dict] = Field(
        default={"api_key": api_key, "base_url": base_url}
    )
    is_azure: bool = Field(default=False)

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_request_count: int = 0

    def __init__(self, max_retry: int = 3, **kwargs):
        args = OpenAIChatArgs()
        args = args.dict()
        client_args = {"api_key": api_key, "base_url": base_url}
        # check if api_key is an azure key
        is_azure = False
        if AZURE_API_KEY and not OPENAI_API_KEY:
            is_azure = True
        for k, v in args.items():
            args[k] = kwargs.pop(k, v)
        if len(kwargs) > 0:
            logger.warn(f"Unused arguments: {kwargs}")
        if args["model"] in LOCAL_LLMS:
            if args["model"] in LOCAL_LLMS_MAPPING:
                client_args["api_key"] = LOCAL_LLMS_MAPPING[args["model"]]["api_key"]
                client_args["base_url"] = LOCAL_LLMS_MAPPING[args["model"]]["base_url"]
                is_azure = False
            else:
                raise ValueError(
                    f"Model {args['model']} not found in LOCAL_LLMS_MAPPING"
                )
        super().__init__(
            args=args, max_retry=max_retry, client_args=client_args, is_azure=is_azure
        )

    @classmethod
    def send_token_limit(self, model: str, base_url: str | None = None) -> int:
        send_token_limit_dict = {
            "gpt-3.5-turbo": 4096,
            "gpt-35-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-3.5-turbo-0613": 16384,
            "gpt-3.5-turbo-1106": 16384,
            "gpt-3.5-turbo-0125": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-0613": 32768,
            "gpt-4-1106-preview": 131072,
            "gpt-4-0125-preview": 131072,
            "llama-2-7b-chat-hf": 4096,
            "gpt-oss-120b": 131072,
            "gpt-oss-120b-131072": 131072,
            "openai/gpt-oss-120b": 131072,
            "openai/gpt-oss-20b": 131072,
            "meta-llama/Meta-Llama-3.1-70B-Instruct": 32768,
            "meta-llama/Meta-Llama-3.1-8B-Instruct": 32768,
            "meta-llama/Llama-3.3-70B-Instruct": 32768,
            "meta-llama/Llama-4-Scout-17B-16E-Instruct": 32768,
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 32768,
            "google/gemma-3-27b-it": 32768,
            "google/gemma-4-26B-A4B-it": 131072,
            "google/gemma-4-31B-it": 131072,
            "google/gemma-4-E4B-it": 131072,
        }
        limit = send_token_limit_dict[model] if model in send_token_limit_dict else 4096
        if _is_quirked_endpoint(base_url) and model in {
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
        }:
            # Some deployments enforce a 16k total window for these
            # Llama 3.1 instruct endpoints, even though the generic model-family
            # default is larger.
            return min(limit, 16384)
        return limit

    def _safe_request_args(self, messages: List[dict]) -> Dict:
        request_args = self.args.dict()
        model = str(request_args.get("model", "") or self.args.model)
        model = _normalize_endpoint_model_name(
            model, base_url=self.client_args.get("base_url")
        )
        request_args["model"] = model
        model_limit = self.send_token_limit(
            model, base_url=self.client_args.get("base_url")
        )
        prompt_tokens = count_message_tokens(messages, model)
        if model in {
            "gpt-oss-120b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "gpt-oss-120b-131072",
        }:
            safety_margin = 2048
        else:
            safety_margin = 256
        available_completion = max(1, model_limit - prompt_tokens - safety_margin)
        requested_max_tokens = int(request_args.get("max_tokens", 0) or 0)

        if prompt_tokens >= model_limit:
            logger.warn(
                f"Prompt for model {model} is estimated at {prompt_tokens} tokens, "
                f"which meets or exceeds the configured context limit {model_limit}. "
                "History should be reduced or prompts shortened."
            )

        if requested_max_tokens < 1 or requested_max_tokens > available_completion:
            logger.warn(
                f"Clamping max_tokens for model {model}: requested={requested_max_tokens}, "
                f"available_completion={available_completion}, prompt_tokens={prompt_tokens}, "
                f"model_limit={model_limit}."
            )
            request_args["max_tokens"] = available_completion

        return request_args

    # @retry(
    #     stop=stop_after_attempt(20),
    #     wait=wait_exponential(multiplier=1, min=4, max=10),
    #     reraise=True,
    #     retry=retry_if_exception_type(
    #         exception_types=(OpenAIError, json.decoder.JSONDecodeError, Exception)
    #     ),
    # )
    def generate_response(
        self,
        prepend_prompt: str = "",
        history: List[dict] = [],
        append_prompt: str = "",
        functions: List[dict] = [],
    ) -> LLMResult:
        messages = self.construct_messages(prepend_prompt, history, append_prompt)
        logger.log_prompt(messages)
        for auth_retry in range(2):
            if self.is_azure:
                openai_client = AzureOpenAI(
                    api_key=self.client_args["api_key"],
                    azure_endpoint=self.client_args["base_url"],
                    api_version="2024-02-15-preview",
                )
            else:
                openai_client = OpenAI(
                    api_key=self.client_args["api_key"],
                    base_url=_choose_base_url(
                        self.args.model, self.client_args["base_url"]
                    ),
                )
            try:
                # Execute function call
                if functions != []:
                    request_args = self._safe_request_args(messages)
                    self.total_request_count += 1
                    response = call_chat_completion_with_retry(
                        openai_client,
                        tag=f"sync:{self.args.model}",
                        messages=messages,
                        functions=functions,
                        **request_args,
                    )

                    logger.log_prompt(
                        [
                            {
                                "role": "assistant",
                                "content": response.choices[0].message.content,
                            }
                        ]
                    )
                    if response.choices[0].message.function_call is not None:
                        self.collect_metrics(response)

                        return LLMResult(
                            content=response.choices[0].message.get("content", ""),
                            function_name=response.choices[0].message.function_call.name,
                            function_arguments=ast.literal_eval(
                                response.choices[0].message.function_call.arguments
                            ),
                            send_tokens=response.usage.prompt_tokens,
                            recv_tokens=response.usage.completion_tokens,
                            total_tokens=response.usage.total_tokens,
                        )
                    else:
                        self.collect_metrics(response)
                        logger.log_prompt(
                            {
                                "role": "assistant",
                                "content": response.choices[0].message.content,
                            }
                        )
                        content = _get_content_or_empty(
                            response, tag=f"sync:{self.args.model}", logger_obj=_MONITOR_LOG
                        )
                        _MONITOR_LOG.info(f"[LLM RAW][sync:{self.args.model}] {repr(content[:5000])}")
                        tag = f"sync:{self.args.model}"   # or sync:...
                        _log_llm_raw(content, tag=tag, logger_obj=_MONITOR_LOG, also_print=True)

                        return LLMResult(
                            content=content,
                            send_tokens=response.usage.prompt_tokens,
                            recv_tokens=response.usage.completion_tokens,
                            total_tokens=response.usage.total_tokens,
                        )

                else:
                    request_args = self._safe_request_args(messages)
                    self.total_request_count += 1
                    response = call_chat_completion_with_retry(
                        openai_client,
                        tag=f"sync:{self.args.model}",
                        messages=messages,
                        **request_args,
                    )
                    logger.log_prompt(
                        [
                            {
                                "role": "assistant",
                                "content": response.choices[0].message.content,
                            }
                        ]
                    )
                    self.collect_metrics(response)
                    content = _get_content_or_empty(
                        response, tag=f"sync:{self.args.model}", logger_obj=_MONITOR_LOG
                    )
                    _MONITOR_LOG.info(f"[LLM RAW][sync:{self.args.model}] {repr(content[:5000])}")
                    tag = f"sync:{self.args.model}"   # or sync:...
                    _log_llm_raw(content, tag=tag, logger_obj=_MONITOR_LOG, also_print=True)

                    return LLMResult(
                        content=content,
                        send_tokens=response.usage.prompt_tokens,
                        recv_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )
            except (OpenAIError, KeyboardInterrupt, json.decoder.JSONDecodeError) as error:
                if isinstance(error, KeyboardInterrupt):
                    raise
                if isinstance(error, json.decoder.JSONDecodeError):
                    raise
                if (
                    auth_retry == 0
                    and _is_auth_error(error, base_url=self.client_args.get("base_url"))
                    and _refresh_access_token(self.client_args, logger_obj=_MONITOR_LOG)
                ):
                    logger.warning(
                        "Auth error detected in sync request; refreshed token and retrying once."
                    )
                    continue
                raise

    # @retry(
    #     stop=stop_after_attempt(20),
    #     wait=wait_exponential(multiplier=1, min=4, max=10),
    #     reraise=True,
    #     retry=retry_if_exception_type(
    #         exception_types=(OpenAIError, json.decoder.JSONDecodeError, Exception)
    #     ),
    # )
    async def agenerate_response(
        self,
        prepend_prompt: str = "",
        history: List[dict] = [],
        append_prompt: str = "",
        functions: List[dict] = [],
    ) -> LLMResult:
        messages = self.construct_messages(prepend_prompt, history, append_prompt)
        logger.log_prompt(messages)

        print("[DEBUG] agenerate_response reached", flush=True)

        for auth_retry in range(2):
            if self.is_azure:
                client_instance = AsyncAzureOpenAI(
                    api_key=self.client_args["api_key"],
                    azure_endpoint=self.client_args["base_url"],
                    api_version="2024-02-15-preview",
                )
            else:
                client_instance = AsyncOpenAI(
                    api_key=self.client_args["api_key"],
                    base_url=_choose_base_url(
                        self.args.model, self.client_args["base_url"]
                    ),
                )

            async with client_instance as async_openai_client:
                try:
                    if functions != []:
                        request_args = self._safe_request_args(messages)
                        self.total_request_count += 1
                        response = await acall_chat_completion_with_retry(
                            async_openai_client,
                            tag=f"async:{self.args.model}",
                            messages=messages,
                            functions=functions,
                            **request_args,
                        )
                        logger.log_prompt(
                            [
                                {
                                    "role": "assistant",
                                    "content": response.choices[0].message.content,
                                }
                            ]
                        )
                        if response.choices[0].message.function_call is not None:
                            function_name = response.choices[0].message.function_call.name
                            valid_function = False
                            if function_name.startswith("function."):
                                function_name = function_name.replace("function.", "")
                            elif function_name.startswith("functions."):
                                function_name = function_name.replace("functions.", "")
                            for function in functions:
                                if function["name"] == function_name:
                                    valid_function = True
                                    break
                            if not valid_function:
                                logger.warn(
                                    f"The returned function name {function_name} is not in the list of valid functions. Retrying..."
                                )
                                raise ValueError(
                                    f"The returned function name {function_name} is not in the list of valid functions."
                                )
                            try:
                                arguments = ast.literal_eval(
                                    response.choices[0].message.function_call.arguments
                                )
                            except:
                                try:
                                    arguments = ast.literal_eval(
                                        JsonRepair(
                                            response.choices[0].message.function_call.arguments
                                        ).repair()
                                    )
                                except:
                                    logger.warn(
                                        "The returned argument in function call is not valid json. Retrying..."
                                    )
                                    raise ValueError(
                                        "The returned argument in function call is not valid json."
                                    )
                            self.collect_metrics(response)
                            logger.log_prompt(
                                {
                                    "role": "assistant",
                                    "content": response.choices[0].message.content,
                                }
                            )
                            content = _get_content_or_empty(
                                response, tag=f"async:{self.args.model}", logger_obj=_MONITOR_LOG
                            )
                            _MONITOR_LOG.info(f"[LLM RAW][async:{self.args.model}] {repr(content[:5000])}")
                            tag = f"async:{self.args.model}"   # or sync:...
                            _log_llm_raw(content, tag=tag, logger_obj=_MONITOR_LOG, also_print=True)

                            return LLMResult(
                                content=content,
                                send_tokens=response.usage.prompt_tokens,
                                recv_tokens=response.usage.completion_tokens,
                                total_tokens=response.usage.total_tokens,
                            )

                        else:
                            self.collect_metrics(response)
                            logger.log_prompt(
                                {
                                    "role": "assistant",
                                    "content": response.choices[0].message.content,
                                }
                            )
                            content = _get_content_or_empty(
                                response, tag=f"async:{self.args.model}", logger_obj=_MONITOR_LOG
                            )
                            _MONITOR_LOG.info(f"[LLM RAW][async:{self.args.model}] {repr(content[:5000])}")
                            tag = f"async:{self.args.model}"   # or sync:...
                            _log_llm_raw(content, tag=tag, logger_obj=_MONITOR_LOG, also_print=True)

                            return LLMResult(
                                content=content,
                                send_tokens=response.usage.prompt_tokens,
                                recv_tokens=response.usage.completion_tokens,
                                total_tokens=response.usage.total_tokens,
                            )

                    else:
                        request_args = self._safe_request_args(messages)
                        self.total_request_count += 1
                        response = await acall_chat_completion_with_retry(
                            async_openai_client,
                            tag=f"async:{self.args.model}",
                            messages=messages,
                            **request_args,
                        )
                        self.collect_metrics(response)
                        logger.log_prompt(
                            [
                                {
                                    "role": "assistant",
                                    "content": response.choices[0].message.content,
                                }
                            ]
                        )
                        content = _get_content_or_empty(
                            response, tag=f"async:{self.args.model}", logger_obj=_MONITOR_LOG
                        )
                        _MONITOR_LOG.info(f"[LLM RAW][async:{self.args.model}] {repr(content[:5000])}")

                        return LLMResult(
                            content=content,
                            send_tokens=response.usage.prompt_tokens,
                            recv_tokens=response.usage.completion_tokens,
                            total_tokens=response.usage.total_tokens,
                        )
                except (OpenAIError, KeyboardInterrupt, json.decoder.JSONDecodeError) as error:
                    if isinstance(error, KeyboardInterrupt):
                        raise
                    if isinstance(error, json.decoder.JSONDecodeError):
                        raise
                    if (
                        auth_retry == 0
                        and _is_auth_error(error, base_url=self.client_args.get("base_url"))
                        and _refresh_access_token(self.client_args, logger_obj=_MONITOR_LOG)
                    ):
                        logger.warning(
                            "Auth error detected in async request; refreshed token and retrying once."
                        )
                        continue
                    raise

    def construct_messages(
        self, prepend_prompt: str, history: List[dict], append_prompt: str
    ):
        messages = []
        if prepend_prompt != "":
            messages.append({"role": "system", "content": prepend_prompt})
        if len(history) > 0:
            messages += history
        if append_prompt != "":
            messages.append({"role": "user", "content": append_prompt})
        return _collapse_consecutive_same_role(messages)

    def collect_metrics(self, response):
        self.total_prompt_tokens += response.usage.prompt_tokens
        self.total_completion_tokens += response.usage.completion_tokens

    def get_spend(self) -> int:
        input_cost_map = {
            "gpt-3.5-turbo": 0.0015,
            "gpt-3.5-turbo-16k": 0.003,
            "gpt-3.5-turbo-0613": 0.0015,
            "gpt-3.5-turbo-16k-0613": 0.003,
            "gpt-3.5-turbo-1106": 0.0005,
            "gpt-3.5-turbo-0125": 0.0005,
            "gpt-4": 0.03,
            "gpt-4-0613": 0.03,
            "gpt-4-32k": 0.06,
            "gpt-4-1106-preview": 0.01,
            "gpt-4-0125-preview": 0.01,
            "llama-2-7b-chat-hf": 0.0,
            # 👉 open-weight models served through an OpenAI-compatible endpoint
            "Llama-4-Maverick-17B-128E-Instruct": 0.0,
            "gpt-oss-120b": 0.0,
            "gpt-oss-120b-131072": 0.0,
            "openai/gpt-oss-120b": 0.0,
            "openai/gpt-oss-20b": 0.0,
            "meta-llama/Meta-Llama-3.1-70B-Instruct": 0.0,
            "meta-llama/Meta-Llama-3.1-8B-Instruct": 0.0,
            "meta-llama/Llama-3.3-70B-Instruct": 0.0,
            "meta-llama/Llama-4-Scout-17B-16E-Instruct": 0.0,
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 0.0,
            "google/gemma-3-27b-it": 0.0,
            "google/gemma-4-26B-A4B-it": 0.0,
            "google/gemma-4-31B-it": 0.0,
            "google/gemma-4-E4B-it": 0.0,

        }

        output_cost_map = {
            "gpt-3.5-turbo": 0.002,
            "gpt-3.5-turbo-16k": 0.004,
            "gpt-3.5-turbo-0613": 0.002,
            "gpt-3.5-turbo-16k-0613": 0.004,
            "gpt-3.5-turbo-1106": 0.0015,
            "gpt-3.5-turbo-0125": 0.0015,
            "gpt-4": 0.06,
            "gpt-4-0613": 0.06,
            "gpt-4-32k": 0.12,
            "gpt-4-1106-preview": 0.03,
            "gpt-4-0125-preview": 0.03,
            "llama-2-7b-chat-hf": 0.0,
            # 👉 open-weight models served through an OpenAI-compatible endpoint
            "Llama-4-Maverick-17B-128E-Instruct": 0.0,
            "gpt-oss-120b": 0.0,
            "gpt-oss-120b-131072": 0.0,
            "openai/gpt-oss-120b": 0.0,
            "openai/gpt-oss-20b": 0.0,
            "meta-llama/Meta-Llama-3.1-70B-Instruct": 0.0,
            "meta-llama/Meta-Llama-3.1-8B-Instruct": 0.0,
            "meta-llama/Llama-3.3-70B-Instruct": 0.0,
            "meta-llama/Llama-4-Scout-17B-16E-Instruct": 0.0,
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct": 0.0,
            "google/gemma-3-27b-it": 0.0,
            "google/gemma-4-26B-A4B-it": 0.0,
            "google/gemma-4-31B-it": 0.0,
            "google/gemma-4-E4B-it": 0.0,
            

        }

        model = self.args.model
        if model not in input_cost_map or model not in output_cost_map:
            logger.warn(
                f"Model type {model} not found in spend table; defaulting spend to $0. "
                "Add it to input_cost_map/output_cost_map if explicit pricing is needed."
            )
            return 0.0

        return (
            self.total_prompt_tokens * input_cost_map[model] / 1000.0
            + self.total_completion_tokens * output_cost_map[model] / 1000.0
        )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def get_embedding(text: str, attempts=3) -> np.array:
    if AZURE_API_KEY and AZURE_API_BASE:
        client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_API_BASE,
            api_version="2024-02-15-preview",
        )
    elif OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    try:
        text = text.replace("\n", " ")
        embedding = client.embeddings.create(
            input=text, model="text-embedding-ada-002"
        ).model_dump_json(indent=2)
        return tuple(embedding)
    except Exception as e:
        attempt += 1
        logger.error(f"Error {e} when requesting openai models. Retrying")
        raise
