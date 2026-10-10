"""Routing policies (spec §6.2). All produce known sampling probabilities."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Uniform:
    """R0: uniform random. probs = 1/n_unlabeled for queued items."""

    budget: int = 300
    seed: int = 0
    id: str = "R0"

    def select(self, unlabeled_ids: list[str], n: int, round_no: int = 0) -> dict[str, float]:
        """Return {item_id: sampling probability} for the next batch of n."""
        rng = random.Random(f"{self.seed}-{round_no}")
        pool = list(unlabeled_ids)
        rng.shuffle(pool)
        chosen = pool[:n]
        p = 1.0 / max(len(unlabeled_ids), 1)
        return {cid: p for cid in chosen}


@dataclass
class Active:
    """R2/R3 placeholder: active sampling by uncertainty/disagreement.

    Phase-1 behavior: falls back to uniform (documented). Real signal weights
    land in Phase 2 behind the probability floor.
    """

    signal: str = "disagreement"
    budget: int = 300
    seed: int = 0
    prob_floor: float = 0.1
    id: str = "R3"

    def select(self, unlabeled_ids: list[str], n: int, round_no: int = 0) -> dict[str, float]:
        rng = random.Random(f"{self.seed}-{round_no}")
        pool = list(unlabeled_ids)
        rng.shuffle(pool)
        chosen = pool[:n]
        p = 1.0 / max(len(unlabeled_ids), 1)
        floor = self.prob_floor * self.budget / max(len(unlabeled_ids), 1) if unlabeled_ids else 0.0
        return {cid: max(p, floor) for cid in chosen}
