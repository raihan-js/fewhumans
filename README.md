# FewHumans

Combine **few human labels** with **many LLM-judge labels** and still report an
**unbiased estimate with a valid confidence interval** — plus which items humans
should label next, and when to stop.

> Status (2026-10-10): Phase 0/1 scaffold. Toy end-to-end works with a fake judge.
> No estimator is validated yet: nothing here is a statistical claim until the
> Phase 4 coverage gate passes. Judge-only intervals are included to measure their
> failure, never as the default.

## 20-line quick start

```python
import fewhumans as fh

task = fh.Task(label_space=fh.Categorical(["correct", "incorrect"]),
               estimand=fh.Mean(of=lambda y: y == "correct"))
panel = fh.Panel([fh.FakeJudge("fake", accuracy=0.8, seed=0)])
s = fh.Session.create("runs/my_eval", items=items, task=task, panel=panel,
                      routing=fh.routing.Uniform(budget=300), confidence=0.95)
s.run_panel()
for item in s.queue(n=20):
    s.add_human(item.item_id, label=truth[item.item_id], annotator="raihan")
est = s.estimate()
print(est)  # accuracy 0.812 [0.789, 0.836], 20 human labels (toy numbers)
```

## CLI

```
fewhumans init runs/x --items items.jsonl --task task.yaml --panel panel.yaml
fewhumans judge runs/x
fewhumans queue runs/x --n 20 > to_label.jsonl
fewhumans add runs/x labels.jsonl
fewhumans estimate runs/x
fewhumans report runs/x --html report.html
```

## What this does not show

- No coverage claim yet (Phase 4 gate pending).
- Toy judges only; real LLM judges land in Phase 3.
- Anytime-valid intervals are stubs behind `experimental=True`.

## Reproduce

```
pip install -e ".[dev]"
pytest -q
python examples/toy_end_to_end.py
```

Plan: `docs/PLAN.md`. Rules: `CLAUDE.md`. Decisions: `docs/decisions.md`.
