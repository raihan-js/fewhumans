"""Judge execution: async, rate-limited, cached, cost-capped (spec §5.4).

- Transports: OpenAI-compatible (vLLM, Ollama, API providers) + Anthropic.
- Cache on disk by (judge config hash, item hash): re-runs are free.
- Cost + latency recorded per call from token counts × an editable price table
  (prices are NEVER hard-coded as facts; they come from study/configs/prices.yaml).
- Strict output schema {"label": ..., "confidence": ...}; parse failures become
  ABSTAIN with the raw text kept, never silently coerced.
- Temperature 0 by default; self_consistency(k) uses temperature > 0 and records
  all samples. A per-run cost cap stops the run (partial results are saved).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


class CostCapExceeded(RuntimeError):
    pass


class JudgeConfig(BaseModel):
    name: str
    provider: str = "openai_compatible"  # openai_compatible | anthropic | ollama
    model: str = ""
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    think: bool = False  # ollama native only: disable thinking (JSON mode needs it)
    prompt_template: str = "Rate the following. Reply with JSON {\"label\": ..., \"confidence\": ...}."
    temperature: float = 0.0
    confidence: str = "verbalized"  # verbalized | logprob | self_consistency
    self_consistency_k: int = 5
    max_tokens: int = 256

    def digest(self) -> str:
        raw = self.model_dump_json(exclude={"api_key"})
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class PriceTable(BaseModel):
    per_1k_in: dict[str, float] = Field(default_factory=dict)
    per_1k_out: dict[str, float] = Field(default_factory=dict)

    def cost(self, model: str, tin: int, tout: int) -> float:
        return tin / 1000 * self.per_1k_in.get(model, 0.0) + tout / 1000 * self.per_1k_out.get(model, 0.0)


class JudgeCall(BaseModel):
    item_id: str
    label: Any = "ABSTAIN"
    confidence: float | None = None
    raw: str = ""
    parse_status: str = "ok"
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    samples: list[str] = Field(default_factory=list)


def _cache_path(cache_dir: Path, cfg: JudgeConfig, item_id: str) -> Path:
    return cache_dir / cfg.digest() / f"{hashlib.sha256(item_id.encode()).hexdigest()[:16]}.json"


def parse_output(text: str) -> tuple[Any, float | None, str]:
    """Strict parse of {"label","confidence"}; failure -> (ABSTAIN, None, raw)."""
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        obj = json.loads(text[start:end])
        return obj.get("label", "ABSTAIN"), obj.get("confidence"), "ok"
    except (ValueError, AttributeError):
        return "ABSTAIN", None, "ABSTAIN"


class Runner:
    def __init__(
        self,
        prices: PriceTable,
        cache_dir: str | Path = "data/cache/judges",
        concurrency: int = 8,
        client: httpx.AsyncClient | None = None,
    ):
        self.prices = prices
        self.cache_dir = Path(cache_dir)
        self.sem = asyncio.Semaphore(concurrency)
        self.client = client or httpx.AsyncClient(timeout=120.0)
        self.spent_usd = 0.0

    def _load_cached(self, cfg: JudgeConfig, item_id: str) -> JudgeCall | None:
        p = _cache_path(self.cache_dir, cfg, item_id)
        if p.exists():
            return JudgeCall(**json.loads(p.read_text()))
        return None

    def _save_cached(self, cfg: JudgeConfig, call: JudgeCall) -> None:
        p = _cache_path(self.cache_dir, cfg, call.item_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(call.model_dump_json())

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1.0, max=30.0), reraise=True)
    async def _post(self, url: str, headers: Mapping[str, str], payload: Mapping[str, Any]) -> httpx.Response:
        r = await self.client.post(url, headers=dict(headers), json=dict(payload))
        r.raise_for_status()
        return r

    async def _complete_openai(self, cfg: JudgeConfig, prompt: str, temperature: float) -> tuple[str, int, int]:
        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": cfg.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        r = await self._post(f"{cfg.base_url}/chat/completions", headers, payload)
        body = r.json()
        choice = body["choices"][0]
        text = choice["message"]["content"] or ""
        usage = body.get("usage", {})
        return text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))

    async def _complete_ollama(self, cfg: JudgeConfig, prompt: str, temperature: float) -> tuple[str, int, int]:
        """Native /api/chat transport. Needed because the OpenAI-compatible
        endpoint ignores think=false and reasoning models return empty content."""
        base = cfg.base_url.rstrip("/v1").rstrip("/")
        payload: dict[str, Any] = {
            "model": cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "think": cfg.think,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": cfg.max_tokens},
        }
        r = await self._post(f"{base}/api/chat", {"Content-Type": "application/json"}, payload)
        body = r.json()
        text = body.get("message", {}).get("content", "") or ""
        return text, int(body.get("prompt_eval_count", 0)), int(body.get("eval_count", 0))

    async def _complete(self, cfg: JudgeConfig, prompt: str, temperature: float) -> tuple[str, int, int]:
        if cfg.provider == "anthropic":
            return await self._complete_anthropic(cfg, prompt, temperature)
        if cfg.provider == "ollama":
            return await self._complete_ollama(cfg, prompt, temperature)
        return await self._complete_openai(cfg, prompt, temperature)

    async def _complete_anthropic(self, cfg: JudgeConfig, prompt: str, temperature: float) -> tuple[str, int, int]:
        payload: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": cfg.api_key,
            "anthropic-version": "2023-06-01",
        }
        r = await self._post("https://api.anthropic.com/v1/messages", headers, payload)
        body = r.json()
        text = "".join(b.get("text", "") for b in body.get("content", []))
        usage = body.get("usage", {})
        return text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))

    async def judge_one(self, cfg: JudgeConfig, item_id: str, prompt: str, max_cost_usd: float) -> JudgeCall:
        cached = self._load_cached(cfg, item_id)
        if cached is not None:
            return cached
        async with self.sem:
            if self.spent_usd >= max_cost_usd:
                raise CostCapExceeded(f"cap ${max_cost_usd} reached at {item_id}")
            t0 = time.monotonic()
            k = cfg.self_consistency_k if cfg.confidence == "self_consistency" else 1
            temp = 0.7 if cfg.confidence == "self_consistency" else cfg.temperature
            texts: list[str] = []
            tin = tout = 0
            for _ in range(k):
                t, i, o = await self._complete(cfg, prompt, temp)
                texts.append(t)
                tin += i
                tout += o
            if k > 1:
                labs = [parse_output(t)[0] for t in texts]
                top = max(set(map(str, labs)), key=list(map(str, labs)).count)
                agree = sum(str(v) == top for v in labs) / len(labs)
                label: Any = top
                conf: float | None = agree
                status, raw = "ok", texts[0]
            else:
                label, conf, status = parse_output(texts[0])
                raw = texts[0]
            cost = self.prices.cost(cfg.model, tin, tout)
            self.spent_usd += cost
            call = JudgeCall(
                item_id=item_id, label=label, confidence=conf, raw=raw,
                parse_status=status, tokens_in=tin, tokens_out=tout,
                cost_usd=cost, latency_s=time.monotonic() - t0, samples=texts,
            )
            self._save_cached(cfg, call)
            return call

    async def run_all(
        self, cfg: JudgeConfig, items: list[tuple[str, str]], max_cost_usd: float = 5.0
    ) -> list[JudgeCall]:
        """Judge (item_id, prompt) pairs. Raises CostCapExceeded (partials saved)."""
        calls = await asyncio.gather(*[self.judge_one(cfg, i, p, max_cost_usd) for i, p in items])
        return list(calls)
