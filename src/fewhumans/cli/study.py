"""`fh-study` CLI: tables + figures from results/*.json (spec §9, D2)."""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(help="FewHumans study runner/analyzer.")


@app.command()
def report(results: Path = Path("results"), out: Path = Path("results/study_summary.md")) -> None:
    """Generate tables/figures via code only (no invented numbers)."""
    rows = sorted(results.glob("*.json"))
    lines = ["# fh-study report", f"{len(rows)} result files in {results}", ""]
    for r in rows:
        try:
            d = json.loads(r.read_text())
            lines.append(f"- `{r.name}`: `{str(d)[:200]}`")
        except Exception as e:  # noqa: BLE001
            lines.append(f"- `{r.name}`: unreadable ({e})")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    typer.echo(f"report -> {out} ({len(rows)} files)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
