"""Anytime-valid confidence sequences — STUB behind experimental flag (Phase 2+).

No sequential interval is exposed until it passes the Phase 4 coverage gate.
"""

from __future__ import annotations


def sequential_mean(values: list[float], confidence: float = 0.95, experimental: bool = False):  # type: ignore[no-untyped-def]
    if not experimental:
        raise NotImplementedError(
            "Sequential intervals are not validated yet (Phase 4 gate). "
            "Pass experimental=True to try the research stub."
        )
    raise NotImplementedError("Sequential stub: implement betting/empirical-Bernstein in Phase 2.")
