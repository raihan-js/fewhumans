"""Phase 4 coverage gate (spec §8): does every default interval cover at nominal rate?

For every (task, method, budget): R Monte Carlo replicates; each replicate
draws human labels per the routing policy from the hidden truth (judgebank
humans), computes the interval with the library estimator, and checks whether
it contains the full-human value. Sequential methods are evaluated at a
simulated stopping time; "peeking" with fixed-sample intervals measures the
inflation (Finding F2). Synthetic stress cases probe robustness.

Usage:
  .venv/bin/python study/simulate/coverage.py --R 2000 --jobs 8        # main gate
  .venv/bin/python study/simulate/coverage.py --only sequential,peeking,stress
  .venv/bin/python study/simulate/coverage.py --aggregate              # rebuild tables

Parts are written to results/coverage_parts/ (resume-safe); the aggregate
writes results/coverage.json, results/coverage_failures.json,
results/sequential.json, results/peeking.json, results/stress.json.
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
    ppi_mean,
    ppi_plus_mean,
    stratified_ppi_plus,
)
from fewhumans.estimators.stratified import neyman_allocation
from fewhumans.routing import vote_entropy
from fewhumans.sequential import cs_human_only

BUDGETS = [50, 100, 200, 400]
METHODS = ["E1+R0", "E2+R0", "E3+R0", "E4+R1", "E5+R0", "E5+R3", "E5+R5"]
CONF = 0.95
FLOOR_FRAC = 0.1
BUDGET_REF = 400  # reference budget for the Poisson floor

# Known value bounds per task for the anytime-valid CS (fixed pools: the
# min/max below come from the task definition or the full human array).
TASK_BOUNDS = {
    "helpsteer2-correctness": (0.0, 4.0),
    "summeval-consistency": (1.0, 5.0),
    "mtbench-winrate": (0.0, 1.0),
    "arena-winrate": (0.0, 1.0),
    "wmt-mqm": None,  # set from the full-human range at runtime
    "toxicity-flag": (0.0, 1.0),
    "oraclebench-truth": (0.0, 1.0),
}


def load_task(task: str) -> dict[str, np.ndarray | list[str]]:
    """Hidden truth + panel predictions from the judgebank (simulator-owned)."""
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
    kind = TASK_KIND[task]
    jm: dict[str, list[float | None]] = {
        j: [map_judge(kind, None, v) for v in vals] for j, vals in per_judge.items()
    }
    # Drop items where any judge failed to parse (analysis-time ABSTAIN).
    keep = [i for i in range(len(ids)) if all(m[i] is not None for m in jm.values())]
    panel = np.array([np.mean([float(m[i]) for m in jm.values()]) for i in keep])
    if kind in ("pairwise", "binary"):
        raws = [[str(per_judge[j][i]) for j in per_judge] for i in keep]
        dis = np.array([vote_entropy(r) for r in raws])
    else:
        mat = np.array([[float(m[i]) for m in jm.values()] for i in keep])
        dis = mat.std(axis=1)
    strata = np.array(["split" if d > 1e-12 else "unanimous" for d in dis])
    return {
        "y": np.array([humans[i] for i in keep]),
        "f": panel,
        "dis": dis,
        "strata": strata,
        "ids": [ids[i] for i in keep],
    }


def poisson_probs(w: np.ndarray, target: int, n: int) -> np.ndarray:
    s = w.sum()
    w = w if s > 0 else np.ones_like(w)
    floor = FLOOR_FRAC * BUDGET_REF / max(n, 1)
    return np.minimum(1.0, np.maximum(floor, target * w / w.sum()))


def draw_sample(task: dict, policy: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (labeled_idx, pi) with known inclusion probabilities."""
    N = len(task["y"])
    if policy == "R0":
        idx = rng.choice(N, min(n, N), replace=False)
        return idx, np.full(len(idx), min(n, N) / N)
    if policy == "R1":
        strata = task["strata"]
        alloc = neyman_allocation(
            {h: int((strata == h).sum()) for h in ("unanimous", "split")}, None, n
        )
        idx_list, pi_list = [], []
        for h, m in alloc.items():
            pool = np.where(strata == h)[0]
            m = min(m, len(pool))
            sel = rng.choice(pool, m, replace=False) if m else np.array([], dtype=int)
            idx_list.append(sel)
            pi_list.append(np.full(len(sel), m / max(len(pool), 1)))
        return np.concatenate(idx_list), np.concatenate(pi_list)
    if policy in ("R3", "R5"):
        w = 0.05 + task["dis"]
        if policy == "R5":
            w = 0.5 * 1.0 + 0.5 * (0.05 + task["dis"]) / 1.05
        pi = poisson_probs(w, n, N)
        draw = rng.random(N) < pi
        idx = np.where(draw)[0]
        return idx, pi[idx]
    raise ValueError(policy)


def estimate(task: dict, method: str, idx: np.ndarray, pi: np.ndarray) -> tuple[float, float, float]:
    y_all = task["y"]
    f_all = task["f"]
    m = method.split("+")[0]
    if m == "E1":
        r = human_only(list(y_all[idx]))
    elif m == "E2":
        r = ppi_mean(list(f_all), list(y_all[idx]), list(f_all[idx]))
    elif m == "E3":
        mask = np.zeros(len(y_all), bool)
        mask[idx] = True
        r = ppi_plus_mean(list(f_all[~mask]), list(y_all[idx]), list(f_all[idx]))
    elif m == "E4":
        lab = {f"i{k}": float(y_all[k]) for k in idx}
        r = stratified_ppi_plus(
            {f"i{k}": str(s) for k, s in enumerate(task["strata"])},
            {f"i{k}": float(f) for k, f in enumerate(f_all)},
            lab,
        )
    elif m == "E5":
        r = active_mean(
            {f"i{k}": float(f_all[k]) for k in range(len(y_all))},
            {f"i{k}": float(y_all[k]) for k in idx},
            {f"i{k}": float(p) for k, p in zip(idx, pi)},
        )
    else:
        raise ValueError(method)
    return r.value, r.lo, r.hi


def run_config(task_name: str, method: str, n: int, R: int, seed: int) -> dict:
    task = load_task(task_name)
    truth = float(task["y"].mean())
    policy = method.split("+")[1]
    rng = np.random.default_rng(seed)
    covers, hws, ns = 0, [], []
    infeasible = 0
    for _ in range(R):
        idx, pi = draw_sample(task, policy, n, rng)
        if len(idx) < 2:
            infeasible += 1
            continue
        try:
            _, lo, hi = estimate(task, method, idx, pi)
        except ValueError:
            infeasible += 1
            continue
        covers += lo <= truth <= hi
        hws.append((hi - lo) / 2)
        ns.append(len(idx))
    c = covers / max(R - infeasible, 1)
    return {
        "task": task_name, "method": method, "n": n, "R": R,
        "coverage": c, "mcse": float(np.sqrt(c * (1 - c) / max(R - infeasible, 1))),
        "mean_hw": float(np.mean(hws)) if hws else None,
        "mean_n": float(np.mean(ns)) if ns else None,
        "infeasible": infeasible,
    }


def target_hw(task_name: str) -> float:
    """Stopping target: half-width of an E1 n=400 sample (analytic, documented)."""
    from scipy import stats as _stats

    task = load_task(task_name)
    y = task["y"]
    t = float(_stats.t.ppf(0.975, 399))
    return float(t * y.std(ddof=1) / np.sqrt(400))


def run_sequential(task_name: str, R: int, seed: int, mode: str) -> dict:
    """Simulate check-after-every-label stopping at target_hw.

    mode=seq uses the anytime-valid CS (E6); mode=peek re-checks the fixed E2
    interval (measures inflation = Finding F2).
    """
    task = load_task(task_name)
    truth = float(task["y"].mean())
    target = target_hw(task_name)
    lo_b, hi_b = TASK_BOUNDS[task_name] or (float(task["y"].min()), float(task["y"].max()))
    rng = np.random.default_rng(seed)
    covers, stops = 0, []
    cap = min(800, len(task["y"]))
    for _ in range(R):
        order = rng.permutation(len(task["y"]))
        stopped = None
        for t in range(10, cap + 1):
            idx = order[:t]
            if mode == "seq":
                r = cs_human_only(list(task["y"][idx]), lo=lo_b, hi=hi_b)
            else:
                r = ppi_mean(list(task["f"]), list(task["y"][idx]), list(task["f"][idx]))
            if r.halfwidth <= target:
                stopped = (t, r.lo <= truth <= r.hi)
                break
        if stopped is None:
            idx = order[:cap]
            r = cs_human_only(list(task["y"][idx]), lo=lo_b, hi=hi_b) if mode == "seq" else ppi_mean(
                list(task["f"]), list(task["y"][idx]), list(task["f"][idx]))
            stopped = (cap, r.lo <= truth <= r.hi)
        stops.append(stopped[0])
        covers += stopped[1]
    c = covers / R
    return {
        "task": task_name, "mode": mode, "R": R, "target_hw": target,
        "coverage": c, "mcse": float(np.sqrt(c * (1 - c) / R)),
        "mean_stop_n": float(np.mean(stops)),
    }


def stress_case(case: str, R: int, seed: int) -> dict:
    """Synthetic judges: S1 easy/hard, S2 class-biased, S3 miscalibrated conf."""
    rng = np.random.default_rng(seed)
    N = 3000
    truth = (rng.random(N) < 0.6).astype(float)
    hard = rng.random(N) < 0.15
    if case == "S1":
        right = np.where(hard, rng.random(N) < 0.30, rng.random(N) < 0.99)
        f = np.where(right, truth, 1 - truth)
    elif case == "S2":
        f = (rng.random(N) < 0.90).astype(float)  # biased to class 1
    elif case == "S3":
        right = rng.random(N) < 0.75
        f = np.where(right, truth, 1 - truth)
    else:
        raise ValueError(case)
    out = {}
    for method in ("E1+R0", "E2+R0", "E3+R0", "E5+R0"):
        covers = 0
        for _ in range(R):
            idx = rng.choice(N, 100, replace=False)
            y, fl = truth[idx], f[idx]
            if method.startswith("E1"):
                r = human_only(list(y))
            elif method.startswith("E2"):
                r = ppi_mean(list(f), list(y), list(fl))
            elif method.startswith("E3"):
                mask = np.zeros(N, bool)
                mask[idx] = True
                r = ppi_plus_mean(list(f[~mask]), list(y), list(fl))
            else:
                r = active_mean(
                    {f"i{k}": float(f[k]) for k in range(N)},
                    {f"i{k}": float(truth[k]) for k in idx},
                    {f"i{k}": 100 / N for k in idx},
                )
            covers += r.lo <= truth.mean() <= r.hi
        c = covers / R
        out[method] = {"coverage": c, "mcse": float(np.sqrt(c * (1 - c) / R))}
    return {"case": case, "R": R, **out}


def part_path(kind: str, key: str) -> Path:
    return Path("results/coverage_parts") / f"{kind}_{key}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", default="main,sequential,peeking,stress")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--tasks", default="helpsteer2-correctness,summeval-consistency,mtbench-winrate,arena-winrate,wmt-mqm,toxicity-flag,oraclebench-truth")
    args = ap.parse_args()
    only = set(args.only.split(","))
    Path("results/coverage_parts").mkdir(parents=True, exist_ok=True)

    if args.aggregate or "main" in only:
        import multiprocessing as mp

        tasks = args.tasks.split(",")
        jobs = []
        for t in tasks:
            for m in METHODS:
                for n in BUDGETS:
                    key = f"{t}_{m}_{n}"
                    if part_path("main", key).exists() and not args.aggregate:
                        continue
                    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                    jobs.append((t, m, n, args.R, seed))
        if jobs and not args.aggregate:
            print(f"running {len(jobs)} configs x R={args.R} on {args.jobs} workers")
            with mp.Pool(args.jobs) as pool:
                for res in pool.starmap(run_config, jobs):
                    part_path("main", f"{res['task']}_{res['method']}_{res['n']}").write_text(json.dumps(res))
                    print(f"  {res['task']} {res['method']} n={res['n']}: cov={res['coverage']:.3f}±{res['mcse']:.3f} hw={res['mean_hw'] and round(res['mean_hw'],4)}", flush=True)
        # Aggregate.
        rows = [json.loads(p.read_text()) for p in sorted(Path("results/coverage_parts").glob("main_*.json"))]
        Path("results/coverage.json").write_text(json.dumps(rows, indent=2))
        fails = [r for r in rows if r["coverage"] < 0.95 - 2 * r["mcse"] and r["infeasible"] < args.R]
        Path("results/coverage_failures.json").write_text(json.dumps(fails, indent=2))
        print(f"aggregate: {len(rows)} configs, {len(fails)} failures (pass rule: cov ≥ 0.95 − 2·MCSE)")

    for kind in ("sequential", "peeking"):
        if kind not in only:
            continue
        mode = "seq" if kind == "sequential" else "peek"
        out = {}
        for t in args.tasks.split(","):
            p = part_path(kind, t)
            if p.exists():
                out[t] = json.loads(p.read_text())
                continue
            res = run_sequential(t, args.R, int(hashlib.sha256(t.encode()).hexdigest()[:8], 16), mode)
            p.write_text(json.dumps(res))
            out[t] = res
            print(f"  {kind} {t}: cov={res['coverage']:.3f}±{res['mcse']:.3f} stop_n={res['mean_stop_n']:.0f}", flush=True)
        Path(f"results/{kind}.json").write_text(json.dumps(out, indent=2))

    if "stress" in only:
        out = {}
        for case in ("S1", "S2", "S3"):
            p = part_path("stress", case)
            if p.exists():
                out[case] = json.loads(p.read_text())
                continue
            res = stress_case(case, args.R, int(hashlib.sha256(case.encode()).hexdigest()[:8], 16))
            p.write_text(json.dumps(res))
            out[case] = res
            print(f"  stress {case}: " + str({m: round(v['coverage'], 3) for m, v in res.items() if m != 'case' and m != 'R'}), flush=True)
        Path("results/stress.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
