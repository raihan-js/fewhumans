"""Phase 5 main study (spec §9.2): savings at equal precision + secondaries.

For every (task, method, panel, budget): R replicates of human-label draws,
library estimates, interval widths, coverage. Judges' panel predictions are
fixed (judgebank); only human labels are simulated. Resume-safe parts in
results/study_parts/ (rep-level widths live there; gitignored). Aggregate:
results/study.json (small, committed).

Methods/panels (gate-validated ranges enforced; E5+R3-tox100 kept with its
known ~1pp caveat, noted in preregistration):
  E1+R0 (—), E0 (full/best/cheap), E2/E3/E5R0 × (full/best/cheap),
  E4+R1 (full), E5+R3 (full), E5+R5 (full).
Panels: full = mean of 3 judges; best = api-fast; cheap = local-qwen35-4b.

Usage:
  .venv/bin/python study/simulate/study.py --R 2000 --jobs 8
  .venv/bin/python study/simulate/study.py --aggregate
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "analyze"))
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from checkpoint3 import TASK_KIND, map_judge

from fewhumans.estimators import (
    active_mean,
    human_only,
    judge_only,
    ppi_mean,
    ppi_plus_mean,
    stratified_ppi_plus,
)

import coverage as gate

GRID = [25, 50, 75, 100, 150, 200, 300, 400, 600]
PANELS = ["full", "best", "cheap"]
JUDGE_OF = {"best": "apifast", "cheap": "4b"}
METHODS = ["E1+R0", "E0", "E2+R0", "E3+R0", "E5+R0", "E4+R1", "E5+R3", "E5+R5"]
CONF = 0.95


def in_range(task: str, method: str, n: int) -> bool:
    m = method.split("+")[0]
    if m == "E4" and n < 100:
        return False
    if task == "wmt-mqm":
        if m in ("E1", "E2") and n < 100:
            return False
        if m in ("E3", "E4") and n < 200:
            return False
    return True


def load_panels(task: str) -> dict:
    """Per-judge mapped predictions + humans + disagreement + strata."""
    kind = TASK_KIND[task]
    per_judge: dict[str, list] = {}
    humans: list[float] = []
    ids: list[str] = []
    for path in sorted(glob.glob(f"data/judgebank/{task}-*.parquet")):
        if "fedproc" in path:
            continue
        df = pd.read_parquet(path)
        j = Path(path).stem.rsplit("-", 1)[1]
        per_judge[j] = list(df["judge"])
        if not ids:
            ids = list(df["item_id"])
            humans = [float(v) for v in df["human"]]
    jm = {j: [map_judge(kind, None, v) for v in vals] for j, vals in per_judge.items()}
    keep = [i for i in range(len(ids)) if all(m[i] is not None for m in jm.values())]
    F = {j: np.array([float(m[i]) for i in keep]) for j, m in jm.items()}
    full = np.mean(list(F.values()), axis=0)
    y = np.array([humans[i] for i in keep])
    if kind in ("pairwise", "binary"):
        from fewhumans.routing import vote_entropy

        raws = [[str(per_judge[j][i]) for j in per_judge] for i in keep]
        dis = np.array([vote_entropy(r) for r in raws])
    else:
        mat = np.array([[float(m[i]) for m in jm.values()] for i in keep])
        dis = mat.std(axis=1)
    return {
        "y": y, "F": F, "full": full, "dis": dis,
        "strata": np.array(["split" if d > 1e-12 else "unanimous" for d in dis]),
        "N": len(keep),
    }


def estimate_cfg(task: dict, method: str, panel: str, idx: np.ndarray, pi: np.ndarray):
    y_all = task["y"]
    F = task["F"]
    f_all = task["full"] if panel == "full" else F[JUDGE_OF[panel]]
    m = method.split("+")[0]
    if m == "E1":
        return human_only(list(y_all[idx]))
    if m == "E0":
        return judge_only(list(f_all))
    if m == "E2":
        return ppi_mean(list(f_all), list(y_all[idx]), list(f_all[idx]))
    if m == "E3":
        mask = np.zeros(len(y_all), bool)
        mask[idx] = True
        return ppi_plus_mean(list(f_all[~mask]), list(y_all[idx]), list(f_all[idx]))
    if m == "E4":
        return stratified_ppi_plus(
            {f"i{k}": str(s) for k, s in enumerate(task["strata"])},
            {f"i{k}": float(f) for k, f in enumerate(f_all)},
            {f"i{k}": float(y_all[k]) for k in idx},
        )
    if m == "E5":
        return active_mean(
            {f"i{k}": float(f_all[k]) for k in range(len(y_all))},
            {f"i{k}": float(y_all[k]) for k in idx},
            {f"i{k}": float(p) for k, p in zip(idx, pi)},
        )
    raise ValueError(method)


def run_config(task_name: str, method: str, panel: str, n: int, R: int, seed: int) -> dict:
    task = load_panels(task_name)
    N = task["N"]
    n = min(n, N)
    truth = float(task["y"].mean())
    policy = method.split("+")[1] if "+" in method else "R0"  # E0 ignores sampling
    rng = np.random.default_rng(seed)
    # Reuse gate sampling on a compatible view.
    view = {"y": task["y"], "f": task["full"], "dis": task["dis"], "strata": task["strata"]}
    hws, covers, ns = [], 0, []
    infeasible = 0
    for _ in range(R):
        idx, pi = gate.draw_sample(view, policy, n, rng)
        if len(idx) < 2:
            infeasible += 1
            continue
        try:
            r = estimate_cfg(task, method, panel, idx, pi)
        except ValueError:
            infeasible += 1
            continue
        hws.append(r.halfwidth)
        covers += r.lo <= truth <= r.hi
        ns.append(len(idx))
    denom = max(R - infeasible, 1)
    return {
        "task": task_name, "method": method, "panel": panel, "n": n, "R": R,
        "mean_hw": float(np.mean(hws)) if hws else None,
        "se_hw": float(np.std(hws, ddof=1) / np.sqrt(len(hws))) if len(hws) > 1 else None,
        "coverage": covers / denom, "mean_n": float(np.mean(ns)) if ns else None,
        "infeasible": infeasible, "hws": [float(x) for x in hws],
    }


def part_path(task: str, method: str, panel: str, n: int) -> Path:
    return Path("results/study_parts") / f"{task}_{method}_{panel}_{n}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--tasks", default="helpsteer2-correctness,summeval-consistency,mtbench-winrate,arena-winrate,wmt-mqm,toxicity-flag,oraclebench-truth")
    args = ap.parse_args()
    Path("results/study_parts").mkdir(parents=True, exist_ok=True)
    tasks = args.tasks.split(",")

    if not args.aggregate:
        import multiprocessing as mp

        jobs = []
        for t in tasks:
            N = load_panels(t)["N"]
            for method in METHODS:
                panels = ["full"] if method in ("E1+R0", "E4+R1", "E5+R3", "E5+R5") else PANELS
                if method == "E0":
                    panels = PANELS
                for n in GRID:
                    if n > N or not in_range(t, method, n):
                        continue
                    for panel in panels:
                        if part_path(t, method, panel, n).exists():
                            continue
                        seed = int(hashlib.sha256(f"{t}{method}{panel}{n}".encode()).hexdigest()[:8], 16)
                        jobs.append((t, method, panel, n, args.R, seed))
        print(f"running {len(jobs)} study configs x R={args.R} on {args.jobs} workers")
        with mp.Pool(args.jobs) as pool:
            for res in pool.starmap(run_config, jobs):
                part_path(res["task"], res["method"], res["panel"], res["n"]).write_text(json.dumps(res))
                print(f"  {res['task']} {res['method']}/{res['panel']} n={res['n']}: hw={res['mean_hw'] and round(res['mean_hw'],4)} cov={round(res['coverage'],3)}", flush=True)

    rows = []
    for p in sorted(Path("results/study_parts").glob("*.json")):
        rows.append(json.loads(p.read_text()))
    small = [{k: v for k, v in r.items() if k != "hws"} for r in rows]
    Path("results/study.json").write_text(json.dumps(small, indent=2))
    print(f"aggregate: {len(rows)} study configs -> results/study.json")


if __name__ == "__main__":
    main()
