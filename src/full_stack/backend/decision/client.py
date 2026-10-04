"""
HTTP transport for structured decision models.

Two routes are supported:

- OpenRouter (default): POST https://openrouter.ai/api/alpha/decisions with the
  OpenRouter key. The response carries the exact USD cost of the call.
- TypeSafe native: POST https://api.typesafe.ai/v1/systemone with TYPESAFE_API_KEY.

Both take the same body: {"model", "state", "questions"} and return
{"model", "answers", "usage"}.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from ..config.settings import get_settings
from .registry import get_decision_model_spec, native_model_id

logger = logging.getLogger("compass.decision.client")

_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
_TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"


class DecisionRequestError(RuntimeError):
    """A decision request failed after retries, or was rejected by the provider."""

    def __init__(self, message: str, *, status: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body

    @property
    def looks_like_length_error(self) -> bool:
        text = f"{self} {self.body}".lower()
        markers = (
            "context length",
            "context window",
            "maximum context",
            "too long",
            "too many tokens",
            "token limit",
            "exceeds the limit",
            "exceeds the maximum",
        )
        return self.status in (400, 413, 422) and any(m in text for m in markers)


@dataclass
class DecisionUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None


@dataclass
class DecisionResponse:
    model: str
    answers: Dict[str, Dict[str, Any]]
    usage: DecisionUsage
    latency_ms: int
    request_id: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


def _openrouter_decisions_url(base_url: str) -> str:
    override = os.getenv("COMPASS_DECISION_ENDPOINT", "").strip()
    if override:
        return override
    base = str(base_url or "https://openrouter.ai/api/v1").rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return f"{base}/alpha/decisions"


class DecisionClient:
    """Sends one state plus a map of typed questions to a decision model."""

    def __init__(self, settings=None, session: Optional[requests.Session] = None):
        self.settings = settings or get_settings()
        self.session = session or requests.Session()
        self.calls: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ routing
    def _route(self, model: str) -> Dict[str, Any]:
        provider = str(getattr(self.settings.decision, "provider", "openrouter") or "openrouter").lower()
        if provider == "typesafe":
            key = str(getattr(self.settings, "typesafe_api_key", "") or "")
            if not key:
                raise DecisionRequestError("TYPESAFE_API_KEY is not set for the TypeSafe decision route.")
            return {"url": _TYPESAFE_URL, "key": key, "model": native_model_id(model), "provider": "typesafe"}
        key = str(getattr(self.settings, "openrouter_api_key", "") or "")
        if not key:
            raise DecisionRequestError("OPENROUTER_API_KEY is not set for the OpenRouter decision route.")
        spec = get_decision_model_spec(model)
        model_id = spec.model_id if spec is not None else str(model)
        return {
            "url": _openrouter_decisions_url(self.settings.openrouter_base_url),
            "key": key,
            "model": model_id,
            "provider": "openrouter",
        }

    # --------------------------------------------------------------------- call
    def ask(self, *, model: str, state: Any, questions: Dict[str, Dict[str, Any]]) -> DecisionResponse:
        if not questions:
            raise ValueError("A decision request needs at least one question.")
        route = self._route(model)
        headers = {"Authorization": f"Bearer {route['key']}", "Content-Type": "application/json"}
        if route["provider"] == "openrouter":
            if self.settings.openrouter_site_url:
                headers["HTTP-Referer"] = self.settings.openrouter_site_url
            if self.settings.openrouter_app_name:
                headers["X-Title"] = self.settings.openrouter_app_name
        body = {"model": route["model"], "state": state, "questions": questions}
        timeout = float(getattr(self.settings.decision, "request_timeout_seconds", 120.0) or 120.0)
        retries = getattr(self.settings.decision, "max_retries", 3)
        attempts = max(1, int(3 if retries is None else retries))

        last_error: Optional[DecisionRequestError] = None
        for attempt in range(1, attempts + 1):
            started = time.time()
            try:
                resp = self.session.post(route["url"], headers=headers, data=json.dumps(body), timeout=timeout)
            except requests.RequestException as exc:
                last_error = DecisionRequestError(f"Decision request failed: {exc}")
                self._sleep(attempt, None)
                continue
            latency_ms = int((time.time() - started) * 1000)
            if resp.status_code == 200:
                try:
                    payload = resp.json()
                except ValueError:
                    last_error = DecisionRequestError(
                        "Decision response was not JSON.", status=200, body=resp.text[:2000]
                    )
                    self._sleep(attempt, None)
                    continue
                answers = payload.get("answers") if isinstance(payload, dict) else None
                if not isinstance(answers, dict):
                    raise DecisionRequestError("Decision response has no answers map.", status=200, body=resp.text[:2000])
                usage_raw = payload.get("usage") or {}
                usage = DecisionUsage(
                    input_tokens=int(usage_raw.get("input_tokens") or 0),
                    output_tokens=int(usage_raw.get("output_tokens") or 0),
                    cost_usd=(float(usage_raw["cost"]) if usage_raw.get("cost") is not None else None),
                )
                spec = get_decision_model_spec(model)
                if usage.cost_usd is None and spec is not None:
                    usage.cost_usd = usage.input_tokens * spec.input_price_per_million / 1e6
                result = DecisionResponse(
                    model=str(payload.get("model") or route["model"]),
                    answers=answers,
                    usage=usage,
                    latency_ms=latency_ms,
                    request_id=str(payload.get("id") or resp.headers.get("X-Generation-Id") or ""),
                    raw=payload,
                )
                self.calls.append(
                    {
                        "model": result.model,
                        "provider": route["provider"],
                        "question_count": len(questions),
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "cost_usd": usage.cost_usd,
                        "latency_ms": latency_ms,
                        "request_id": result.request_id,
                    }
                )
                print(
                    f"[DecisionClient] {result.model}: {len(questions)} questions, "
                    f"{usage.input_tokens} input tokens, {latency_ms} ms"
                )
                return result

            text = resp.text[:2000]
            last_error = DecisionRequestError(
                f"Decision request returned HTTP {resp.status_code}: {text[:300]}",
                status=resp.status_code,
                body=text,
            )
            if resp.status_code not in _RETRY_STATUS:
                raise last_error
            self._sleep(attempt, resp.headers.get("retry-after"))

        raise last_error or DecisionRequestError("Decision request failed.")

    @staticmethod
    def _sleep(attempt: int, retry_after: Optional[str]) -> None:
        try:
            delay = float(retry_after) if retry_after else 0.0
        except ValueError:
            delay = 0.0
        delay = max(delay, min(20.0, 1.5 * (2 ** (attempt - 1))))
        time.sleep(delay)

    def total_cost(self) -> float:
        return float(sum(float(c.get("cost_usd") or 0.0) for c in self.calls))
