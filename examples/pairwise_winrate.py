"""Which of two systems wins? Pairwise judgments + disagreement routing.

Scenario: judges compared responses from system A vs B on 600 prompts
(A/B/tie). Truth: A wins 58% (ties count half). You can afford 80 humans.
Route them by panel disagreement (R3) and estimate the win rate with E5.

Usage:
    pip install fewhumans
    python examples/pairwise_winrate.py --n-human 80 --outdir runs/pairwise
"""

from __future__ import annotations

import argparse
from pathlib import Path

import fewhumans as fh
from fewhumans.core.items import Item
from fewhumans.core.session import Session
from fewhumans.core.task import Categorical, Mean, Task

W2N = {"A": 1.0, "B": 0.0, "tie": 0.5}


def main(n_items: int = 600, n_human: int = 80, seed: int = 0, outdir: str = "runs/pairwise") -> dict:
    import random

    rng = random.Random(seed)
    truth = {}
    for k in range(n_items):
        r = rng.random()
        truth[f"i{k}"] = "A" if r < 0.50 else ("tie" if r < 0.66 else "B")  # A wins 58%
    items = [Item(iid, {"pair": iid, "truth": t}) for iid, t in truth.items()]

    class VoteJudge(fh.Judge):
        def __init__(self, name: str, acc: float, seed: int):
            self.name = name
            self.acc = acc
            self.seed = seed

        def label(self, item: Item) -> fh.JudgeOutput:
            r = random.Random(hash((self.seed, item.item_id)) & 0xFFFFFFFF)
            t = item.payload["truth"]
            if r.random() < self.acc:
                return fh.JudgeOutput(item_id=item.item_id, label=t, confidence=0.9, raw=t)
            wrong = [v for v in ("A", "B", "tie") if v != t]
            w = wrong[int(r.random() * len(wrong))]
            return fh.JudgeOutput(item_id=item.item_id, label=w, confidence=0.6, raw=w)

    task = Task(Categorical(["A", "B", "tie"]), Mean(of=lambda y: W2N[str(y)]))
    panel = fh.Panel([VoteJudge("v1", 0.75, 1), VoteJudge("v2", 0.70, 2)])
    out = Path(outdir)
    s = Session.create(out / "session", items, task, panel,
                       fh.routing.Disagreement(budget=300, seed=seed))
    s.run_panel()
    got = 0
    while got < n_human:
        q = s.queue(n=min(20, n_human - got))
        if not q:
            break
        for it in q:
            s.add_human(it.item_id, label=truth[it.item_id], annotator="you")
        got += len(q)
    e1 = s.estimate("E1-human-only")
    e5 = s.estimate("E5-active")
    print("truth (hidden in practice): 0.580")
    print(f"E1 human-only ({got} labels): {e1}")
    print(f"E5 active ({got} labels):     {e5}")
    s.report(out / "report.html")
    return {"e1": e1.to_dict(), "e5": e5.to_dict(), "n": got}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=600)
    ap.add_argument("--n-human", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="runs/pairwise")
    a = ap.parse_args()
    main(a.n_items, a.n_human, a.seed, a.outdir)
