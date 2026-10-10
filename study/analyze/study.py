"""Phase 5 analysis (spec §9.3): savings ratios, pooling, calibration, costs.

Reads results/study_parts/*.json (rep-level widths) + judgebank parquets.
Writes results/study_summary.json, results/forest.png, results/calibration.png.
Targets w and hypotheses frozen in docs/preregistration.md BEFORE the study ran.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

W = {  # frozen targets = analytic E1 half-width at n_ref (see preregistration)
    "helpsteer2-correctness": 0.1217,
    "summeval-consistency": 0.4420,
    "mtbench-winrate": 0.0500,
    "arena-winrate": 0.0412,
    "wmt-mqm": 0.2947,
    "toxicity-flag": 0.0492,
    "oraclebench-truth": 0.0412,
}
HUMAN_USD = 0.30
HUMAN_RANGE = (0.10, 1.00)


def load_parts() -> dict:
    out: dict = {}
    for p in sorted(glob.glob("results/study_parts/*.json")):
        r = json.loads(Path(p).read_text())
        out.setdefault(r["task"], {}).setdefault(f"{r['method']}/{r['panel']}", {})[r["n"]] = r
    return out


def curve_ns(task_curves: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    ns = np.array(sorted(task_curves[key]))
    hws = np.array([task_curves[key][n]["mean_hw"] for n in ns])
    # Enforce non-increasing (running minimum from the left) for interpolation.
    dec = np.minimum.accumulate(hws)
    return ns.astype(float), dec


def n_to_target(ns: np.ndarray, hws: np.ndarray, w: float) -> float:
    """Labels needed for mean-hw ≤ w (linear interp; censored at grid ends)."""
    if hws[0] <= w:
        return float(ns[0]) / 2  # already below at smallest grid point
    if hws[-1] > w:
        return float(ns[-1]) * 2  # never reaches within grid
    # hws decreasing: find segment
    for i in range(len(ns) - 1):
        if hws[i] >= w >= hws[i + 1]:
            frac = (hws[i] - w) / max(hws[i] - hws[i + 1], 1e-12)
            return float(ns[i] + frac * (ns[i + 1] - ns[i]))
    return float(ns[-1])


def savings_bootstrap(task_curves: dict, key: str, base: str, w: float, B: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    R = len(next(iter(task_curves[key].values()))["hws"])
    ratios = []
    for _ in range(B):
        idx = rng.integers(0, R, R)
        def boot_hw(k: str, n: int) -> float:  # noqa: ANN202
            h = np.array(task_curves[k][n]["hws"])[idx]
            return float(h.mean())
        ns_k = np.array(sorted(task_curves[key]))
        ns_b = np.array(sorted(task_curves[base]))
        hk = np.minimum.accumulate(np.array([boot_hw(key, n) for n in ns_k]))
        hb = np.minimum.accumulate(np.array([boot_hw(base, n) for n in ns_b]))
        ratios.append(n_to_target(ns_k, hk, w) / max(n_to_target(ns_b, hb, w), 1e-9))
    ratios = np.array(ratios)
    return {"ratio": float(np.median(ratios)), "lo": float(np.quantile(ratios, 0.025)), "hi": float(np.quantile(ratios, 0.975))}


def dl_pool(ests: list[float], ses: list[float]) -> dict:
    """DerSimonian-Laird random-effects pool of log-ratios."""
    y = np.array(ests)
    v = np.array(ses) ** 2 + 1e-12
    w_fe = 1 / v
    mu_fe = (w_fe * y).sum() / w_fe.sum()
    Q = (w_fe * (y - mu_fe) ** 2).sum()
    df = len(y) - 1
    tau2 = max(0.0, (Q - df) / (w_fe.sum() - (w_fe**2).sum() / w_fe.sum())) if df > 0 else 0.0
    w_re = 1 / (v + tau2)
    mu = (w_re * y).sum() / w_re.sum()
    se = float(np.sqrt(1 / w_re.sum()))
    return {"log_ratio": float(mu), "se": se, "tau2": float(tau2),
            "ratio": float(np.exp(mu)), "lo": float(np.exp(mu - 1.96 * se)), "hi": float(np.exp(mu + 1.96 * se))}


def task_props() -> dict:
    """Routing-help analysis inputs: AUROC(dis→wrong), agreement, imbalance."""
    from checkpoint3 import TASK_KIND

    import sys

    sys.path.insert(0, "study/analyze")
    props = {}
    for task, kind in TASK_KIND.items():
        if task == "fedproc-roles":
            continue
        dfs = {}
        for path in sorted(glob.glob(f"data/judgebank/{task}-*.parquet")):
            j = Path(path).stem.rsplit("-", 1)[1]
            dfs[j] = pd.read_parquet(path)
        base = dfs.get("9b", next(iter(dfs.values())))
        y = base["human"].to_numpy(dtype=float)
        # Panel (mean of mapped judges across files).
        from checkpoint3 import map_judge

        F = np.mean([[map_judge(kind, None, v) or 0.0 for v in dfs[j]["judge"]] for j in dfs], axis=0)
        wrong = (np.abs(F - y) > (0.26 if kind in ("binary", "pairwise") else 0.5 * (y.std() or 1))).astype(float)
        # Disagreement proxy: std across judges' mapped values.
        M = np.array([[map_judge(kind, None, v) or 0.0 for v in dfs[j]["judge"]] for j in dfs])
        dis = M.std(axis=0)
        n1, n0 = wrong.sum(), len(wrong) - wrong.sum()
        if n1 == 0 or n0 == 0:
            auroc = float("nan")
        else:
            order = np.argsort(dis, kind="stable")
            ranked = wrong[order]
            auroc = float((np.arange(len(ranked))[ranked == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))
        agree = float(np.mean(np.abs(F - y) < (0.26 if kind in ("binary", "pairwise") else 0.5)))
        props[task] = {"auroc_dis_wrong": auroc, "panel_agreement": agree,
                       "truth_sd": float(y.std()), "kind": kind, "N": len(y)}
    return props


def calibration() -> dict:
    """ECE + reliability bins per judge per task (verbalized confidence)."""
    from checkpoint3 import TASK_KIND, map_judge

    out = {}
    for task, kind in TASK_KIND.items():
        if task == "fedproc-roles":
            continue
        out[task] = {}
        for path in sorted(glob.glob(f"data/judgebank/{task}-*.parquet")):
            j = Path(path).stem.rsplit("-", 1)[1]
            df = pd.read_parquet(path)
            conf = pd.to_numeric(df["confidence"], errors="coerce").to_numpy()
            f = np.array([map_judge(kind, None, v) for v in df["judge"]], dtype=float)
            y = df["human"].to_numpy(dtype=float)
            tol = 0.26 if kind in ("binary", "pairwise") else (0.5 if kind == "ordinal" else y.std() / 2 or 0.5)
            correct = (np.abs(f - y) <= tol).astype(float)
            mask = ~np.isnan(conf)
            conf, correct = conf[mask], correct[mask]
            ece, bins = 0.0, []
            for b in range(10):
                m = (conf >= b / 10) & (conf < (b + 1) / 10 + (b == 9))
                if m.sum() == 0:
                    continue
                ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
                bins.append({"conf": float(conf[m].mean()), "acc": float(correct[m].mean()), "mass": float(m.mean())})
            out[task][j] = {"ece": float(ece), "bins": bins, "n": int(mask.sum())}
    return out


def main() -> None:
    parts = load_parts()
    base = "E1+R0/full"
    summary: dict = {"targets": W, "tasks": {}, "pooled": {}, "props": task_props(), "calibration": calibration()}
    for task, curves in parts.items():
        w = W[task]
        summary["tasks"][task] = {}
        # E1 reference n(w).
        ns_b, hb = curve_ns(curves, base)
        n_base = n_to_target(ns_b, hb, w)
        summary["tasks"][task]["E1_n"] = n_base
        for key in sorted(curves):
            if key == base or key.startswith("E0/"):
                continue
            ns, h = curve_ns(curves, key)
            n_k = n_to_target(ns, h, w)
            sb = savings_bootstrap(curves, key, base, w)
            summary["tasks"][task][key] = {"n": n_k, **sb,
                                           "hw_at": {str(n): curves[key][n]["mean_hw"] for n in sorted(curves[key])},
                                           "coverage": {str(n): curves[key][n]["coverage"] for n in sorted(curves[key])}}
        # E0 judge-only error (H4).
        for key in sorted(curves):
            if key.startswith("E0/"):
                ns0 = sorted(curves[key])
                r0 = curves[key][ns0[0]]
                summary["tasks"][task][key] = {"coverage": r0["coverage"], "mean_hw": r0["mean_hw"]}
    # Pooled savings per method/panel (log-ratio DL).
    methods = sorted({k for t in parts.values() for k in t if not k.startswith("E0/") and k != base})
    for key in methods:
        ests, ses = [], []
        for task, curves in parts.items():
            if key not in curves:
                continue
            s = savings_bootstrap(curves, key, base, W[task])
            if s["lo"] <= 0:
                continue
            ests.append(float(np.log(s["ratio"])))
            ses.append(max((float(np.log(s["hi"])) - float(np.log(s["lo"]))) / (2 * 1.96), 1e-6))
        if ests:
            summary["pooled"][key] = dl_pool(ests, ses)
    Path("results/study_summary.json").write_text(json.dumps(summary, indent=2))

    # Forest plot of pooled savings ratios.
    keys = sorted(summary["pooled"])
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(keys) + 1)))
    ys = np.arange(len(keys))
    for i, k in enumerate(keys):
        p = summary["pooled"][k]
        ax.errorbar(p["ratio"], i, xerr=[[p["ratio"] - p["lo"]], [p["hi"] - p["ratio"]]], fmt="o", capsize=4)
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_yticks(ys, keys, fontsize=8)
    ax.set_xlabel("human labels vs E1+R0 (ratio, <1 = savings)")
    ax.set_title("Pooled human-label savings at equal precision (DL random-effects)")
    fig.tight_layout()
    fig.savefig("results/forest.png", dpi=100)

    # Calibration plot (ECE bars).
    cal = summary["calibration"]
    tasks = sorted(cal)
    judges = sorted({j for t in cal.values() for j in t})
    x = np.arange(len(tasks))
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for k, j in enumerate(judges):
        ax2.bar(x + 0.25 * (k - 1), [cal[t].get(j, {}).get("ece", 0.0) for t in tasks], width=0.25, label=j)
    ax2.set_xticks(x, tasks, rotation=20, fontsize=7)
    ax2.set_ylabel("ECE (verbalized confidence)")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig2.savefig("results/calibration.png", dpi=100)
    print("wrote results/study_summary.json, forest.png, calibration.png")
    for k in keys:
        p = summary["pooled"][k]
        print(f"  {k}: {p['ratio']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}]")


if __name__ == "__main__":
    main()
