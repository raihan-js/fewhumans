"""Routing base."""

from __future__ import annotations

from typing import Any, Protocol


class RoutingPolicy(Protocol):
    id: str

    def select(
        self,
        unlabeled_ids: list[str],
        n: int,
        round_no: int = 0,
        ctx: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        ...
