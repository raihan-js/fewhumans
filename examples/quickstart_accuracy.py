"""Estimate a system's accuracy from judge labels + a few human labels.

Scenario: an LLM judge labeled 1,000 responses as correct/incorrect. You can
afford 50 human labels. What is the system's accuracy, with a valid interval?

Shows E0 (judge-only: biased, invalid) vs E1 (human-only) vs E3 (PPI++),
a stopping check, and the HTML report card.

Usage:
    pip install fewhumans
    python examples/quickstart_accuracy.py --n-human 50 --outdir runs/quickstart
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fewhumans as fh
from fewhumans.core.items import Item
from fewhumans.core.session import Session
from fewhumans.core.task import Categorical, Mean, Task


def main(n_items: int = 1000, n_human: int = 50, seed: int = 0, outdir: str = "runs/quickstart") -> dict:
    import random

    rng = random.Random(seed)
    truth = {f"i{k}": "correct" if rng.random() < 0.72 else "incorrect" for k in range(n_items)}
    items = [Item(iid, {"text": f"response {iid}", "truth": t}) for iid, t in truth.items()]
    task = Task(Categorical(["correct", "incorrect"]), Mean(of=lambda y: str(y) == "correct"))
    panel = fh.Panel([
        fh.FakeJudge("judge-a", accuracy=0.85, seed=1),
        fh.FakeJudge("judge-b", accuracy=0.80, seed=2),
    ])
    out = Path(outdir)
    s = Session.create(out / "session", items, task, panel, fh.routing.Uniform(budget=300))
    s.run_panel()
    for it in s.queue(n=n_human):
        s.add_human(it.item_id, label=truth[it.item_id], annotator="you")
    e0 = s.estimate("E0-judge-only")
    e1 = s.estimate("E1-human-only")
    e3 = s.estimate("E3-ppi-plus")
    print(f"truth (hidden in practice):      {sum(v == 'correct' for v in truth.values()) / n_items:.3f}")
    print(f"E0 judge-only:                   {e0}   <-- covers the JUDGE mean, not truth")
    print(f"E1 human-only ({n_human} labels):          {e1}")
    print(f"E3 PPI++ ({n_human} labels):               {e3}")
    print(s.should_stop(target_halfwidth=0.03)["explanation"])
    s.report(out / "report.html")
    print(f"report -> {out / 'report.html'}")
    return {"e0": e0.to_dict(), "e1": e1.to_dict(), "e3": e3.to_dict()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=1000)
    ap.add_argument("--n-human", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="runs/quickstart")
    a = ap.parse_args()
    main(a.n_items, a.n_human, a.seed, a.outdir)
