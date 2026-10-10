"""Items to be labeled (spec §5.1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Item:
    """An object to be labeled.

    Attributes:
        item_id: stable unique id.
        payload: text / pair / span / path — anything JSON-serializable.
        strata: optional user metadata used for stratified routing.
    """

    item_id: str
    payload: Any
    strata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Item:
        return Item(item_id=str(d["item_id"]), payload=d.get("payload"), strata=dict(d.get("strata") or {}))
