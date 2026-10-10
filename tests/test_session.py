"""Session save/resume + toy end-to-end + hand-picked refusal."""

import pytest

import fewhumans as fh
from fewhumans.core.items import Item
from fewhumans.core.session import Session
from fewhumans.core.task import Categorical, Mean, Task


def _toy(n=60, seed=1):
    rng_items = [Item(f"i{k}", {"text": f"text {k}", "truth": "correct" if k % 2 == 0 else "incorrect"}) for k in range(n)]
    task = Task(Categorical(["correct", "incorrect"]), Mean(of=lambda y: str(y) == "correct"))
    panel = fh.Panel([fh.FakeJudge("fake", accuracy=0.8, seed=seed)])
    return rng_items, task, panel


def test_session_roundtrip(tmp_path):
    items, task, panel = _toy()
    s = Session.create(tmp_path / "s", items, task, panel, fh.routing.Uniform(budget=300))
    s.run_panel()
    q = s.queue(n=10)
    assert len(q) == 10
    for it in q:
        s.add_human(it.item_id, label=it.payload["truth"], annotator="t")
    est = s.estimate()
    assert est.n_human == 10
    s2 = Session.load(tmp_path / "s", task, panel, fh.routing.Uniform(budget=300))
    assert len(s2.humans) == 10
    assert abs(s2.estimate().value - est.value) < 1e-12


def test_handpicked_refused(tmp_path):
    items, task, panel = _toy()
    s = Session.create(tmp_path / "s", items, task, panel, fh.routing.Uniform())
    s.run_panel()
    with pytest.raises(ValueError, match="known probability"):
        s.add_human(items[0].item_id, label="correct")


def test_queued_probs_survive_reload(tmp_path):
    items, task, panel = _toy()
    s = Session.create(tmp_path / "s", items, task, panel, fh.routing.Uniform(budget=300))
    s.run_panel()
    q = s.queue(n=5)
    s2 = Session.load(tmp_path / "s", task, panel, fh.routing.Uniform(budget=300))
    for it in q:  # queued probs persist, so humans can be added after reload
        s2.add_human(it.item_id, label=it.payload["truth"], annotator="t")
    assert len(s2.humans) == 5
