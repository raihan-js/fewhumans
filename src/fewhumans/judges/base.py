"""Judge base classes (spec §5.1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from fewhumans.core.items import Item


@dataclass(frozen=True)
class JudgeOutput:
    item_id: str
    label: Any
    confidence: float | None = None
    raw: str = ""
    parse_status: str = "ok"  # ok | ABSTAIN
    tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Judge:
    """Anything mapping an item to a label + optional confidence."""

    name: str = "judge"

    def label(self, item: Item) -> JudgeOutput:
        raise NotImplementedError

    def label_many(self, items: list[Item]) -> list[JudgeOutput]:
        return [self.label(it) for it in items]

    def config(self) -> dict[str, Any]:
        return {"name": self.name, "type": type(self).__name__}
