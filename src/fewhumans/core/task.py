"""Label spaces, estimands, tasks (spec §5.1)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Binary:
    labels: tuple[str, str] = ("incorrect", "correct")


@dataclass(frozen=True)
class Categorical:
    classes: tuple[str, ...]

    def __init__(self, classes: list[str] | tuple[str, ...]):
        object.__setattr__(self, "classes", tuple(classes))


@dataclass(frozen=True)
class Ordinal:
    lo: int = 1
    hi: int = 5


@dataclass(frozen=True)
class Real:
    pass


@dataclass(frozen=True)
class Mean:
    """Estimand: population mean of a numeric mapping of the label.

    Example: accuracy = Mean(of=lambda y: y == "correct").
    """

    of: Callable[[Any], float]

    def map(self, y: Any) -> float:
        return float(self.of(y))


@dataclass(frozen=True)
class Task:
    label_space: Any
    estimand: Mean
