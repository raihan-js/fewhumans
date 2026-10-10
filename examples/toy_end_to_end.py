"""Toy end-to-end: fake judge, uniform routing, PPI estimate (Checkpoint 1)."""

import fewhumans as fh
from fewhumans.core.items import Item
from fewhumans.core.session import Session
from fewhumans.core.task import Categorical, Mean, Task

items = [Item(f"i{k}", {"text": f"item {k}", "truth": "correct" if k % 3 else "incorrect"}) for k in range(200)]
task = Task(Categorical(["correct", "incorrect"]), Mean(of=lambda y: str(y) == "correct"))
panel = fh.Panel([fh.FakeJudge("fake", accuracy=0.8, seed=0)])
s = Session.create("runs/toy", items, task, panel, fh.routing.Uniform(budget=300), confidence=0.95)
print(s.run_panel())
for it in s.queue(n=20):
    s.add_human(it.item_id, label=it.payload["truth"], annotator="example")
est = s.estimate()
print(est)
print(s.should_stop(target_halfwidth=0.02))
s.report("runs/toy_report.html")
print("report -> runs/toy_report.html")
