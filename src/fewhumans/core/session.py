"""Session: items + panel outputs + routing + humans + estimates (spec §5.1)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fewhumans.core.items import Item
from fewhumans.core.task import Task
from fewhumans.estimators import EstimateResult, human_only, judge_only, ppi_mean
from fewhumans.judges.base import Judge


class Session:
    """Holds one evaluation run. Saved as a folder of JSONL + JSON (Phase 1)."""

    def __init__(
        self,
        path: str | Path,
        items: list[Item],
        task: Task,
        panel: Sequence[Judge],
        routing: Any,
        confidence: float = 0.95,
        sequential: bool = False,
    ):
        self.path = Path(path)
        self.items = items
        self.task = task
        self.panel = list(panel)
        self.routing_policy = routing
        self.confidence = confidence
        self.sequential = sequential
        self.panel_outputs: dict[str, dict[str, Any]] = {}  # item_id -> {judge: label, ...}
        self.panel_conf: dict[str, dict[str, float | None]] = {}
        self.humans: dict[str, Any] = {}  # item_id -> label
        self.probs: dict[str, float] = {}  # item_id -> sampling probability
        self.annotators: dict[str, str] = {}
        self.history: list[dict[str, Any]] = []
        self.round_no = 0

    # -- construction / persistence --
    @staticmethod
    def create(path: str | Path, items: list[Item], task: Task, panel: Sequence[Judge], routing: Any, confidence: float = 0.95, sequential: bool = False) -> Session:
        s = Session(path, items, task, panel, routing, confidence, sequential)
        s.save()
        return s

    @staticmethod
    def load(path: str | Path, task: Task, panel: Sequence[Judge], routing: Any) -> Session:
        p = Path(path)
        items: list[Item] = []
        for line in (p / "items.jsonl").read_text().splitlines():
            if line.strip():
                items.append(Item.from_dict(json.loads(line)))
        s = Session(path, items, task, panel, routing)
        meta = json.loads((p / "session.json").read_text())
        s.confidence = meta.get("confidence", 0.95)
        s.round_no = meta.get("round_no", 0)
        if (p / "panel.jsonl").exists():
            for line in (p / "panel.jsonl").read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    s.panel_outputs[r["item_id"]] = r["labels"]
                    s.panel_conf[r["item_id"]] = r.get("confidences", {})
        if (p / "humans.jsonl").exists():
            for line in (p / "humans.jsonl").read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    s.humans[r["item_id"]] = r["label"]
                    s.probs[r["item_id"]] = r["prob"]
                    s.annotators[r["item_id"]] = r.get("annotator", "")
        if (p / "queue.jsonl").exists():
            for line in (p / "queue.jsonl").read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    s.probs.setdefault(r["item_id"], r["prob"])
        if (p / "history.json").exists():
            s.history = json.loads((p / "history.json").read_text())
        return s

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / "items.jsonl", "w") as f:
            f.writelines(json.dumps(it.to_dict()) + "\n" for it in self.items)
        with open(self.path / "panel.jsonl", "w") as f:
            f.writelines(json.dumps({"item_id": iid, "labels": labs, "confidences": self.panel_conf.get(iid, {})}) + "\n" for iid, labs in self.panel_outputs.items())
        with open(self.path / "humans.jsonl", "w") as f:
            f.writelines(json.dumps({"item_id": iid, "label": lab, "prob": self.probs[iid], "annotator": self.annotators.get(iid, "")}) + "\n" for iid, lab in self.humans.items())
        with open(self.path / "queue.jsonl", "w") as f:
            f.writelines(json.dumps({"item_id": iid, "prob": p}) + "\n" for iid, p in self.probs.items())
        (self.path / "session.json").write_text(json.dumps({
            "confidence": self.confidence, "round_no": self.round_no,
            "panel": [j.config() for j in self.panel],
            "routing": type(self.routing_policy).__name__,
        }, indent=2))
        (self.path / "history.json").write_text(json.dumps(self.history, indent=2))

    # -- panel / routing / humans --
    def run_panel(self, max_cost_usd: float = 5.0) -> dict[str, int]:
        counts = {"labeled": 0}
        for it in self.items:
            if it.item_id in self.panel_outputs:
                continue
            labs, confs = {}, {}
            for j in self.panel:
                out = j.label(it)
                labs[j.name] = out.label
                confs[j.name] = out.confidence
            self.panel_outputs[it.item_id] = labs
            self.panel_conf[it.item_id] = confs
            counts["labeled"] += 1
        self.save()
        return counts

    def _panel_pred(self, item_id: str) -> Any:
        labs = self.panel_outputs.get(item_id, {})
        if not labs:
            return None
        vals = list(labs.values())
        return max(set(map(str, vals)), key=list(map(str, vals)).count)

    def queue(self, n: int = 20) -> list[Item]:
        unlabeled = [it.item_id for it in self.items if it.item_id not in self.humans]
        probs = self.routing_policy.select(unlabeled, n, self.round_no)
        self.round_no += 1
        for iid, p in probs.items():
            self.probs.setdefault(iid, p)
        self.save()
        by_id = {it.item_id: it for it in self.items}
        return [by_id[iid] for iid in probs]

    def add_human(self, item_id: str, label: Any, annotator: str = "") -> None:
        if item_id not in self.probs:
            raise ValueError(
                f"item {item_id} was not sampled with known probability "
                "(hand-picked labels cannot be used for estimation)"
            )
        self.humans[item_id] = label
        self.annotators[item_id] = annotator
        self.save()

    # -- estimation --
    def _mapped(self) -> tuple[list[float], list[float], list[float], list[float]]:
        est = self.task.estimand
        judge_all = [est.map(self._panel_pred(it.item_id)) for it in self.items if self._panel_pred(it.item_id) is not None]
        h_ids = list(self.humans)
        y = [est.map(self.humans[i]) for i in h_ids]
        f = [est.map(self._panel_pred(i)) for i in h_ids]
        return judge_all, y, f, [est.map(self.humans[i]) for i in h_ids]

    def estimate(self, method: str = "auto") -> EstimateResult:
        judge_all, y, f, _ = self._mapped()
        if method == "auto":
            method = "E2-ppi" if (y and judge_all) else ("E1-human-only" if y else "E0-judge-only")
        if method.startswith("E0"):
            res = judge_only(judge_all, self.confidence)
        elif method.startswith("E1"):
            if not y:
                raise ValueError("no human labels yet")
            res = human_only(y, self.confidence)
        elif method.startswith("E2"):
            if not y or not judge_all:
                raise ValueError("PPI needs panel outputs + human labels")
            res = ppi_mean(judge_all, y, f, self.confidence)
        else:
            raise ValueError(f"unknown method {method} (Phase 1 supports E0/E1/E2)")
        self.history.append({"method": res.method, **res.to_dict()})
        self.save()
        return res

    def agreement(self) -> dict[str, Any]:
        """Descriptive agreement on human-labeled items (NOT an estimate)."""
        n = same = 0
        for iid, h in self.humans.items():
            p = self._panel_pred(iid)
            if p is not None:
                n += 1
                same += str(p) == str(h)
        return {"n_compared": n, "agreement": (same / n) if n else None, "note": "descriptive only, not an estimate"}

    def should_stop(self, target_halfwidth: float) -> dict[str, Any]:
        res = self.estimate()
        ok = res.halfwidth <= target_halfwidth
        return {
            "stop": bool(ok),
            "halfwidth": res.halfwidth,
            "target": target_halfwidth,
            "n_human": res.n_human,
            "explanation": f"half-width {res.halfwidth:.4f} vs target {target_halfwidth:.4f} ({res.n_human} labels)",
        }

    def report(self, out_html: str | Path) -> str:
        from fewhumans.report.card import render_html

        res = self.estimate() if (self.humans or self.panel_outputs) else None
        html = render_html(self, res)
        Path(out_html).write_text(html)
        return str(out_html)
