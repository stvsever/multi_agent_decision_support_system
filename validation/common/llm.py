"""
Minimal OpenRouter chat client for validation-side LLM work (ontology building).

This is deliberately self-contained and independent of the engine's own LLM
client so the validation layer can be reasoned about in isolation. It reads
``OPENROUTER_API_KEY`` from the repository-root ``.env`` (the same key the
engine uses) and talks to OpenRouter through the OpenAI-compatible SDK.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import dotenv_values
from openai import OpenAI

# Repository root is three levels up from this file:
# validation/common/llm.py -> validation/common -> validation -> <root>
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _load_key() -> str:
    values = dotenv_values(REPO_ROOT / ".env") or {}
    key = str(values.get("OPENROUTER_API_KEY", "") or "").strip().strip('"').strip("'")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not found in .env. Add it to the repository-root "
            ".env file before running the ontology builder."
        )
    return key

class OntologyLLM:
    """Thin wrapper around an OpenRouter chat model used for ontology work."""

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.2):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(api_key=_load_key(), base_url=OPENROUTER_BASE_URL)

    def chat_json(
        self,
        system: str,
        user: str,
        max_tokens: int = 8000,
        retries: int = 3,
    ) -> Dict[str, Any]:
        """Call the model and parse a single JSON object from the reply.

        The tiny models used for cost-efficient testing occasionally wrap JSON
        in prose or code fences, so we extract the first balanced object.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                text = resp.choices[0].message.content or ""
                return _extract_json_object(text)
            except Exception as exc:  # noqa: BLE001 - surface a clean error after retries
                last_err = exc
                if attempt < retries:
                    time.sleep(1.5 * attempt)
        raise RuntimeError(f"OntologyLLM.chat_json failed after {retries} attempts: {last_err}")


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract the first balanced ``{...}`` JSON object from arbitrary text."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model reply: {text[:200]!r}")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Unbalanced JSON object in model reply")
