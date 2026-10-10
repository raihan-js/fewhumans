"""Checkpoint 2 demo: all estimators × policies on a synthetic task with known truth.

Synthetic setup (seeded): N=5000 binary items, truth ~ Bern(0.65). Two judges:
j_easy is right 92% on easy items but 55% on hard items (20% of pool);
j_disagree casts a correlated second vote. Panel prediction = majority mapped
to {0,1}; disagreement = vote entropy.

For budgets {50, 100, 200, 400} (20 replicates): E1+R0, E2+R0, E3+R0, E4+R1,
E5+R3. Reports mean half-width, estimate RMSE vs truth, and judge-only error.
Outputs: results/phase2_synthetic.json + results/phase2_synthetic.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fewhumans.estimators import (
    active_mean,
    human_only,
    judge_only,
    ppi_mean,
    ppi_plus_mean,
    stratified_ppi_plus,
)

BUDGETS = [50, 100, 200, 400]
REPS = 20
N = 5000


def make_task(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    hard = rng.random(N) < 0.2
    truth = (rng.random(N) < 0.65).astype(float)
    # Judge 1: good on easy, near-random on hard. Judge 2: independent copy.
    right1 = np.where(hard, rng.random(N) < 0.55, rng.random(N) < 0.92)
    right2 = np.where(hard, rng.random(N) < 0.55, rng.random(N) < 0.88)
    j1 = np.where(right1, truth, 1 - truth)
    j2 = np.where(right2, truth, 1 - truth)
    pred = ((j1 + j2) >= 1).astype(float)  # majority of 2 = OR; ties -> 1
    dis = (j1 != j2).astype(float)  # disagreement signal
    strata = np.where(j1 == j2, "unanimous", "split")
    return {"truth": truth, "pred": pred, "dis": dis, "strata": strata}


def run_budget(task: dict[str, np.ndarray], budget: int, rep: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(1000 + rep)
    truth, pred = task["truth"], task["pred"]
    out: dict[str, dict[str, float]] = {}

    def rec(name: str, est: float, hw: float) -> None:
        out[name] = {"est": float(est), "hw": float(hw), "err": float(abs(est - truth.mean()))}

    # E1/E2/E3 + uniform human sample.
    idx = rng.choice(N, budget, replace=False)
    y = truth[idx]
    r = human_only(list(y))
    rec("E1+R0", r.value, r.halfwidth)
    r = ppi_mean(list(pred), list(y), list(pred[idx]))
    rec("E2+R0", r.value, r.halfwidth)
    mask = np.zeros(N, bool)
    mask[idx] = True
    r = ppi_plus_mean(list(pred[~mask]), list(y), list(pred[idx]))
    rec("E3+R0", r.value, r.halfwidth)

    # E4+R1: Neyman-ish stratified sample (proportional here for simplicity).
    strata = task["strata"]
    ids = np.arange(N)
    lab_idx: list[int] = []
    for s in ("unanimous", "split"):
        pool = ids[strata == s]
        m = max(2, int(round(budget * len(pool) / N)))
        lab_idx += list(rng.choice(pool, min(m, len(pool)), replace=False))
    lab_idx = lab_idx[:budget]
    r = stratified_ppi_plus(
        {f"i{k}": str(strata[k]) for k in range(N)},
        {f"i{k}": float(pred[k]) for k in range(N)},
        {f"i{k}": float(truth[k]) for k in lab_idx},
    )
    rec("E4+R1", r.value, r.halfwidth)

    # E5+R3: Poisson sampling ∝ disagreement.
    w = 0.05 + task["dis"]
    pi = np.minimum(1.0, budget * w / w.sum())
    pi = np.maximum(pi, 0.1 * budget / N)
    draw = rng.random(N) < pi
    lab5 = ids[draw]
    if len(lab5) >= 2:
        r = active_mean(
            {f"i{k}": float(pred[k]) for k in range(N)},
            {f"i{k}": float(truth[k]) for k in lab5},
            {f"i{k}": float(pi[k]) for k in lab5},
        )
        rec("E5+R3", r.value, r.halfwidth)
        out["E5+R3"]["n_realized"] = len(lab5)
    else:
        rec("E5+R3", float("nan"), float("nan"))

    # E5+R0: same estimator, uniform sample (separates estimator from routing).
    idx0 = rng.choice(N, budget, replace=False)
    r = active_mean(
        {f"i{k}": float(pred[k]) for k in range(N)},
        {f"i{k}": float(truth[k]) for k in idx0},
        {f"i{k}": float(budget / N) for k in idx0},
    )
    rec("E5+R0", r.value, r.halfwidth)
    return out


def main() -> None:
    task = make_task()
    truth_mean = float(task["truth"].mean())
    r0 = judge_only(list(task["pred"]))
    # AUROC of the disagreement signal for "judge wrong" (§9.3 analysis).
    wrong = (task["pred"] != task["truth"]).astype(float)
    order = np.argsort(task["dis"], kind="stable")
    ranked = wrong[order]
    n1, n0 = ranked.sum(), len(ranked) - ranked.sum()
    auroc = (np.arange(len(ranked))[ranked == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for b in BUDGETS:
        reps = [run_budget(task, b, rep) for rep in range(REPS)]
        methods = sorted({m for rep in reps for m in rep})
        summary[str(b)] = {}
        for m in methods:
            hws = [rep[m]["hw"] for rep in reps if np.isfinite(rep[m]["hw"])]
            errs = [rep[m]["err"] for rep in reps if np.isfinite(rep[m]["err"])]
            summary[str(b)][m] = {
                "mean_hw": float(np.mean(hws)),
                "mean_abs_err": float(np.mean(errs)),
                "reps": len(hws),
            }
    results = {
        "truth": truth_mean,
        "judge_only": {"est": r0.value, "hw": r0.halfwidth, "covers_truth": bool(r0.lo <= truth_mean <= r0.hi)},
        "disagreement_auroc_for_judge_wrong": float(auroc),
        "budgets": summary,
        "note": "E0 interval covers the judge mean, not the truth (covers_truth expected False).",
    }
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "phase2_synthetic.json").write_text(json.dumps(results, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    methods = sorted(next(iter(summary.values())))
    xs = np.arange(len(BUDGETS))
    for m in methods:
        axes[0].plot(xs, [summary[str(b)][m]["mean_hw"] for b in BUDGETS], "o-", label=m)
    axes[0].set_xticks(xs, [str(b) for b in BUDGETS])
    axes[0].set_xlabel("human budget")
    axes[0].set_ylabel("mean half-width")
    axes[0].legend(fontsize=8)
    for m in methods:
        axes[1].plot(xs, [summary[str(b)][m]["mean_abs_err"] for b in BUDGETS], "o-", label=m)
    axes[1].set_xticks(xs, [str(b) for b in BUDGETS])
    axes[1].set_xlabel("human budget")
    axes[1].set_ylabel("mean |est − truth|")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Phase-2 synthetic: truth={truth_mean:.3f}, judge-only={r0.value:.3f}")
    fig.tight_layout()
    fig.savefig(out_dir / "phase2_synthetic.png", dpi=100)
    print(json.dumps(results, indent=2)[:1500])
    print("wrote results/phase2_synthetic.json + .png")


if __name__ == "__main__":
    main()
