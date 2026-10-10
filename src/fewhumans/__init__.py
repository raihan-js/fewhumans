"""FewHumans: few human labels + many judge labels, valid intervals."""

from collections.abc import Sequence

from fewhumans import routing
from fewhumans.core.items import Item
from fewhumans.core.session import Session
from fewhumans.core.task import Binary, Categorical, Mean, Ordinal, Real, Task
from fewhumans.judges.base import Judge, JudgeOutput
from fewhumans.judges.fake import FakeJudge
from fewhumans.judges.python_judge import PythonJudge

__version__ = "1.0.0"

__all__ = [
    "Binary",
    "Categorical",
    "FakeJudge",
    "Item",
    "Judge",
    "JudgeOutput",
    "Mean",
    "Ordinal",
    "PythonJudge",
    "Real",
    "Session",
    "Task",
    "routing",
]


class Panel(list[Judge]):  # list of judges with a name
    """A panel of judges (spec §5.1)."""

    def __init__(self, judges: Sequence[Judge] | None = None) -> None:
        super().__init__(judges or [])
