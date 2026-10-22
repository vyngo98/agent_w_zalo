"""Per-request observability: a structured log record emitted once per chat
request and appended to an append-only JSONL file (`REQUEST_LOG_PATH`).

Kept intentionally simple (one JSON line per request) to match the local,
single-user scope of the project. Several fields (cache_hit, guardrail_*,
cost_usd) have no backend in the current local-Ollama setup, so they carry
honest defaults and act as extension points for when those capabilities exist.
"""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from app.config import REQUEST_LOG_PATH


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RequestLog:
    request_id: str
    user_id: str
    timestamp: str
    prompt_template: str
    prompt_version: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cache_hit: bool
    guardrail_input_pass: bool
    guardrail_output_pass: bool
    cost_usd: float
    error: str | None = None


# Per-1K-token prices keyed by model name. Empty for local Ollama (inference is
# free). Fill this in when a paid provider is added, e.g.
#   "claude-opus-4-8": {"input": 0.015, "output": 0.075}
PRICE_TABLE: dict[str, dict[str, float]] = {}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate request cost in USD. Returns 0.0 for local/unknown models."""
    price = PRICE_TABLE.get(model)
    if not price:
        return 0.0
    return round(
        (input_tokens / 1000.0) * price.get("input", 0.0)
        + (output_tokens / 1000.0) * price.get("output", 0.0),
        6,
    )


def write_request_log(log: RequestLog) -> None:
    """Append one JSON line for `log`. Never raises — a logging failure must not
    break the chat response."""
    try:
        os.makedirs(os.path.dirname(REQUEST_LOG_PATH) or ".", exist_ok=True)
        with open(REQUEST_LOG_PATH, "a") as f:
            f.write(json.dumps(asdict(log), ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - observability must not break the request
        pass
