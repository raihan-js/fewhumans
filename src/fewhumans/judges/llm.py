"""LLM judge stub — real API execution lands in Phase 3."""

from __future__ import annotations

from typing import Any

from fewhumans.core.items import Item
from fewhumans.judges.base import Judge, JudgeOutput


class LLMJudge(Judge):
    """Config holder for Phase 1; .label() raises until Phase 3 wires execution."""

    def __init__(self, name: str, model: str, prompt: str = "", confidence: str = "verbalized", **kw: Any):
        self.name = name
        self.model = model
        self.prompt = prompt
        self.confidence_method = confidence
        self.extra = kw

    def label(self, item: Item) -> JudgeOutput:
        raise NotImplementedError("LLMJudge execution lands in Phase 3 (needs API keys + prompts).")

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "type": "LLMJudge", "model": self.model, "confidence": self.confidence_method}
