"""Checkpoint 3: per task x judge agreement/accuracy, abstain rate, cost.

Reads data/judgebank/*.parquet -> results/checkpoint3.json (printed too).
Every number comes from the parquets; no invented values.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def map_judge(kind: str, labels: list[str] | None, val: object) -> float | None:
    s = str(val)
    if kind in ("ordinal", "real"):
        try:
            return float(s)
        except ValueError:
            return None
    if kind == "pairwise":
        return {"A": 1.0, "B": 0.0, "tie": 0.5}.get(s)
    if kind == "binary":
        low = s.lower()
        if "non-" in low or low in ("incorrect", "no", "0", "false"):
            return 0.0
        if low in ("toxic", "correct", "yes", "1", "true"):
            return 1.0
        return None
    if kind == "categorical" and labels:
        if s in labels:
            return s  # type: ignore[return-value]
        pref = [lab for lab in labels if lab.startswith(s) or s.startswith(lab)]
        return pref[0] if len(pref) == 1 else None
    return None


TASK_KIND = {
    "helpsteer2-correctness": "ordinal",
    "summeval-consistency": "ordinal",
    "mtbench-winrate": "pairwise",
    "arena-winrate": "pairwise",
    "wmt-mqm": "real",
    "toxicity-flag": "binary",
    "oraclebench-truth": "binary",
    "fedproc-roles": "categorical",
}

CAT_LABELS = [
    "INCORPORATED_BY_REFERENCE", "INTERNAL_REFERENCE", "CHECKLIST_NOT_SELECTED",
    "FULL_TEXT", "CHECKLIST_SELECTED", "NARRATIVE_MENTION", "INDEX_ENTRY",
    "NOT_A_CLAUSE", "EXPLICITLY_EXCLUDED",
]

# Weak-truth data (model-consensus gold), excluded from the main study.
DROP = {"fedproc-roles"}


def analyze(task: str, path: str) -> dict[str, object]:
    df = pd.read_parquet(path)
    kind = TASK_KIND[task]
    abstain = float((df["parse_status"] != "ok").mean())
    mapped = [map_judge(kind, CAT_LABELS if kind == "categorical" else None, v) for v in df["judge"]]
    ok = [(m, h) for m, h in zip(mapped, df["human"]) if m is not None]
    out: dict[str, object] = {
        "n": len(df), "abstain_rate": abstain,
        "parsed_rate": len(ok) / max(len(df), 1),
        "cost_usd": float(df["cost_usd"].sum()),
        "mean_latency_s": float(df["latency_s"].median()),
    }
    if not ok:
        return out
    if kind == "categorical":
        out["agreement"] = float(np.mean([m == str(h) for m, h in ok]))
    else:
        m = np.array([float(x) for x, _ in ok])
        h = np.array([float(hh) for (_, hh) in ok])
        out["mae"] = float(np.abs(m - h).mean())
        out["corr"] = float(np.corrcoef(m, h)[0, 1]) if len(ok) > 2 and m.std() > 0 and h.std() > 0 else None
        if kind in ("binary", "pairwise"):
            out["agreement"] = float(np.mean(np.abs(m - h) < 0.26))
    return out


def main() -> None:
    results: dict[str, dict[str, object]] = {}
    for path in sorted(glob.glob("data/judgebank/*.parquet")):
        name = Path(path).stem
        if name.startswith("pilot-"):
            continue
        task, judge = name.rsplit("-", 1)
        if task in DROP:
            continue
        judge = {"9b": "local-qwen35-9b", "4b": "local-qwen35-4b", "apifast": "api-fast"}[judge]
        results.setdefault(task, {})[judge] = analyze(task, path)
    Path("results").mkdir(exist_ok=True)
    Path("results/checkpoint3.json").write_text(json.dumps(results, indent=2))
    for task, judges in results.items():
        print(f"== {task}")
        for j, r in judges.items():
            print(f"  {j}: " + ", ".join(f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
