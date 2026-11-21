"""LLM layer built on LangChain (ChatGroq), quota-aware for the Groq free tier.

- `run_chain(prompt, variables, schema, tier)` runs  prompt | ChatGroq(JSON mode)  and validates the output
  against a Pydantic schema.
- Model pool per tier (e.g. gpt-oss-120b -> qwen3.8-27b -> gpt-oss-20b). Each Groq model has its own
  per-minute (TPM) and per-day (TPD) token quota, so spreading load across models keeps answers LLM-backed.
- Per-minute 429: wait the advised time (up to RATE_WAIT_MAX_S) and retry the same model once.
  Per-day 429: that model is benched until its advised reset time, and the next model is used.
- Returns None when there is no key or every model fails, so the graph's deterministic fallbacks take over.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache

from pydantic import BaseModel, ValidationError

from config import GROQ_API_KEY, LLM_MODEL_BACKUP, LLM_MODEL_FAST, LLM_MODEL_STRONG

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover
    ChatGroq = None

RATE_WAIT_MAX_S = 20.0
_benched: dict[str, float] = {}          # model -> unix time when its daily quota should allow requests again
_bench_lock = threading.Lock()


@dataclass
class LLMStats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)


def available() -> bool:
    return bool(GROQ_API_KEY) and ChatGroq is not None


def _pool(tier: str) -> list[str]:
    order = [LLM_MODEL_STRONG, LLM_MODEL_BACKUP, LLM_MODEL_FAST] if tier == "strong" else [LLM_MODEL_FAST, LLM_MODEL_BACKUP, LLM_MODEL_STRONG]
    return [m for m in dict.fromkeys(order) if m]


@lru_cache(maxsize=16)
def _chat(model: str, temperature: float, max_tokens: int):
    extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
    return ChatGroq(model=model, api_key=GROQ_API_KEY, temperature=temperature, max_tokens=max_tokens,
                    max_retries=0, timeout=60, **extra).bind(response_format={"type": "json_object"})


def _parse_wait(msg: str) -> float | None:
    """Parse Groq's 'Please try again in 1m2.5s' / '850ms' into seconds."""
    m = re.search(r"try again in ((?:\d+h)?(?:\d+m(?!s))?(?:[\d.]+s)?(?:[\d.]+ms)?)", msg)
    if not m:
        return None
    s, total = m.group(1), 0.0
    for val, unit in re.findall(r"([\d.]+)(h|ms|m|s)", s):
        total += float(val) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total


def _is_benched(model: str) -> bool:
    with _bench_lock:
        return _benched.get(model, 0) > time.time()


def quota_status() -> dict[str, str]:
    now = time.time()
    with _bench_lock:
        return {m: (f"daily quota exhausted, retry in ~{int((t - now) / 60) + 1} min" if t > now else "ok")
                for m, t in _benched.items()} | {m: "ok" for m in _pool("strong") if m not in _benched}


def _parse_json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)  # some models emit reasoning tags
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _note(stats: LLMStats | None, msg: str):
    if stats is not None and msg not in stats.errors:
        stats.errors.append(msg)


def run_chain(prompt: "ChatPromptTemplate", variables: dict, schema: type[BaseModel], tier: str = "strong",
              stats: LLMStats | None = None, temperature: float = 0.0, max_tokens: int = 1500) -> BaseModel | None:
    if not available():
        return None
    messages = prompt.format_messages(**variables)
    for model in _pool(tier):
        if _is_benched(model):
            _note(stats, f"{model}: daily quota exhausted - skipped")
            continue
        for attempt in range(2):
            try:
                msg = _chat(model, temperature, max_tokens).invoke(messages)
            except Exception as e:  # noqa: BLE001 - classify provider errors below
                text = str(e)
                if "429" in text or "rate_limit" in text.lower() or "Rate limit" in text:
                    wait = _parse_wait(text) or 5.0
                    if "per day" in text or wait > RATE_WAIT_MAX_S:
                        with _bench_lock:
                            _benched[model] = time.time() + wait
                        _note(stats, f"{model}: {'daily' if 'per day' in text else 'rate'} limit, retry in ~{int(wait)}s - switching model")
                        break
                    if attempt == 0:
                        _note(stats, f"{model}: per-minute limit, waited {wait:.1f}s")
                        time.sleep(wait + 0.3)
                        continue
                    break
                _note(stats, f"{model}: {type(e).__name__}: {text[:140]}")
                break
            if stats is not None:
                stats.calls += 1
                stats.models.append(msg.response_metadata.get("model_name", model))
                if msg.usage_metadata:
                    stats.prompt_tokens += msg.usage_metadata.get("input_tokens", 0)
                    stats.completion_tokens += msg.usage_metadata.get("output_tokens", 0)
            data = _parse_json(msg.content)
            if data is None:
                _note(stats, f"{model}: non-JSON output")
                break
            try:
                return schema.model_validate(data)
            except ValidationError as e:
                _note(stats, f"{model}: schema validation failed: {str(e)[:120]}")
                break
    return None
