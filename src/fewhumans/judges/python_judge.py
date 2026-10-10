"""Python judge: any callable (e.g. a classifier)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fewhumans.core.items import Item
from fewhumans.judges.base import Judge, JudgeOutput


class PythonJudge(Judge):
    def __init__(self, name: str, fn: Callable[[Item], Any], confidence_fn: Callable[[Item], float] | None = None):
        self.name = name
        self.fn = fn
        self.confidence_fn = confidence_fn

    def label(self, item: Item) -> JudgeOutput:
        lab = self.fn(item)
        conf = self.confidence_fn(item) if self.confidence_fn else None
        return JudgeOutput(item_id=item.item_id, label=lab, confidence=conf, raw=str(lab))
