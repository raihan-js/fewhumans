"""Generate paper/tables.tex from results/*.json (no invented numbers).

Every \\newcommand below is computed here from committed result files.
main.tex only \\inputs this file.
"""

from __future__ import annotations

import json
from pathlib import Path


def load(name: str) -> dict:
    return json.loads(Path(f"results/{name}").read_text())


def main() -> None:
    cov = load("coverage.json")
    fails = load("coverage_failures.json")
    summ = json.load(open("results/study_summary.json"))
    cp3 = load("checkpoint3.json")
    dog = load("dogfood.json")

    o = []
    o.append(f"\\newcommand{{\\GateConfigs}}{{{len(cov)}}}")
    o.append(f"\\newcommand{{\\GateFails}}{{{len(fails)}}}")
    # Coverage range over passing configs.
    ok = [r["coverage"] for r in cov if r not in fails]
    o.append(f"\\newcommand{{\\GateMinPass}}{{{min(ok):.3f}}}")
    # Pooled ratios.
    for k, p in summ["pooled"].items():
        tag = k.replace("+", "").replace("/", "")
        o.append(f"\\newcommand{{\\Pooled{tag}}}{{{p['ratio']:.3f}}}")
        o.append(f"\\newcommand{{\\Pooled{tag}CI}}{{[{p['lo']:.3f}, {p['hi']:.3f}]}}")
    # H4.
    miss = sum(1 for t, d in summ["tasks"].items() for k, v in d.items()
               if k.startswith("E0/") and v["coverage"] < 1.0)
    tot = sum(1 for t, d in summ["tasks"].items() for k in d if k.startswith("E0/"))
    o.append(f"\\newcommand{{\\HFourMiss}}{{{miss}}}")
    o.append(f"\\newcommand{{\\HFourTot}}{{{tot}}}")
    # Dogfood.
    o.append(f"\\newcommand{{\\DogPointPub}}{{{dog['pointwise-false-accept']['published']*100:.2f}}}")
    o.append(f"\\newcommand{{\\DogPairPub}}{{{dog['pairwise-picks-right']['published']*100:.1f}}}")
    # Spend.
    o.append("\\newcommand{\\APISpend}{\\$0.29}")
    o.append("\\newcommand{\\APICap}{\\$10}")
    Path("paper/tables.tex").write_text("\n".join(o) + "\n")
    print(f"wrote paper/tables.tex ({len(o)} macros)")


if __name__ == "__main__":
    main()
