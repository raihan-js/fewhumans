"""Fake judge for toy runs and tests (no model needed)."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from fewhumans.core.items import Item
from fewhumans.judges.base import Judge, JudgeOutput


class FakeJudge(Judge):
    """A deterministic fake judge.

    With probability `accuracy` returns truth(item), else returns `wrong_label`.
    Confidence is fixed (0.9 when right, 0.6 when wrong) so routing smoke tests work.
    """

    def __init__(
        self,
        name: str = "fake",
        accuracy: float = 0.8,
        seed: int = 0,
        truth: Callable[[Item], Any] | None = None,
        wrong_label: Any = "incorrect",
    ):
        self.name = name
        self.accuracy = accuracy
        self.seed = seed
        self.truth = truth
        self.wrong_label = wrong_label

    def label(self, item: Item) -> JudgeOutput:
        rng = random.Random(hash((self.seed, item.item_id)) & 0xFFFFFFFF)
        truth = self.truth(item) if self.truth else item.payload.get("truth", "correct") if isinstance(item.payload, dict) else "correct"
        if rng.random() < self.accuracy:
            return JudgeOutput(item_id=item.item_id, label=truth, confidence=0.9, raw=str(truth))
        return JudgeOutput(item_id=item.item_id, label=self.wrong_label, confidence=0.6, raw=str(self.wrong_label))

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "type": "FakeJudge", "accuracy": self.accuracy, "seed": self.seed}
