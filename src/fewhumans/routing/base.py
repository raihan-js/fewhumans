"""Routing base."""

from __future__ import annotations

from typing import Protocol


class RoutingPolicy(Protocol):
    id: str

    def select(self, unlabeled_ids: list[str], n: int, round_no: int = 0) -> dict[str, float]:
        ...
