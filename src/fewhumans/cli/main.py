"""`fewhumans` CLI (spec §5.3, file-based)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from fewhumans.core.session import Session
from fewhumans.core.task import Categorical, Mean, Task
from fewhumans.io.jsonl import read_items_jsonl, read_labels_jsonl
from fewhumans.judges.fake import FakeJudge
from fewhumans.routing import Uniform

app = typer.Typer(help="FewHumans: few human labels, valid estimates.")


def _load_session(path: Path) -> Session:
    # Phase-1: task/panel come from session.json written at init.
    meta = json.loads((path / "session.json").read_text())
    _ = meta
    task = Task(label_space=Categorical(["correct", "incorrect"]),
                estimand=Mean(of=lambda y: (str(y) == "correct")))
    panel = [FakeJudge(j.get("name", "fake"), accuracy=j.get("accuracy", 0.8), seed=j.get("seed", 0))
             for j in meta.get("panel", [{"name": "fake"}])]
    routing = Uniform()
    return Session.load(path, task, panel, routing)


@app.command()
def init(path: Path, items: Path = typer.Option(...), task: Path | None = typer.Option(None), panel: Path | None = typer.Option(None)) -> None:
    """Create a session dir from items JSONL."""
    its = read_items_jsonl(items)
    t = Task(label_space=Categorical(["correct", "incorrect"]),
             estimand=Mean(of=lambda y: (str(y) == "correct")))
    p = [FakeJudge("fake", accuracy=0.8, seed=0)]
    Session.create(path, its, t, p, Uniform())
    typer.echo(f"init: {len(its)} items -> {path}")


@app.command()
def judge(path: Path, max_cost: float = 5.0) -> None:
    s = _load_session(path)
    c = s.run_panel(max_cost_usd=max_cost)
    typer.echo(json.dumps(c))


@app.command()
def queue(path: Path, n: int = 20) -> None:
    s = _load_session(Path(path))
    for it in s.queue(n):
        typer.echo(json.dumps(it.to_dict()))


@app.command()
def add(path: Path, labels: Path) -> None:
    s = _load_session(Path(path))
    rows = read_labels_jsonl(labels)
    for r in rows:
        s.add_human(r["item_id"], r["label"], r.get("annotator", ""))
    typer.echo(f"added {len(rows)} human labels")


@app.command()
def estimate(path: Path) -> None:
    s = _load_session(Path(path))
    res = s.estimate()
    typer.echo(json.dumps(res.to_dict(), indent=2))
    typer.echo(str(res))


@app.command(name="report")
def report_cmd(path: Path, html: Path = typer.Option(Path("report.html"), "--html")) -> None:
    s = _load_session(Path(path))
    out = s.report(html)
    typer.echo(f"report -> {out}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
