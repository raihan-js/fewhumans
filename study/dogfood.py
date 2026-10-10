"""Phase 7 dogfooding (spec §11): re-run OracleBench numbers through FewHumans.

Case 1 (pointwise): qwen3b false-accept rate among truly-wrong answers.
  Published: 147/1333 = 11.0% [9.5, 12.8]. Y = 1{judge accepts} (oracle =
  free deterministic "human"), f = qwen0.5b verdict. N=1,333, n=100 uniform.
Case 2 (pairwise matched, plain): qwen3b picks-right rate.
  Published: 379/471 = 80.5% [76.6, 83.8]. Y = 1{picks right},
  f = qwen0.5b pick. N=471 (parsed), n=100 uniform.

FedProc-Ledger deferred (logged): no per-item judge verdicts published and
gold is model-consensus; queued for the VETR reviewer-gold round.

Writes results/dogfood.json. No new human labels: oracle = deterministic.
"""

from __future__ import annotations

import json

import numpy as np
from huggingface_hub import hf_hub_download

from fewhumans.estimators import human_only, ppi_mean, ppi_plus_mean


def load_pointwise() -> tuple[np.ndarray, np.ndarray]:
    p3 = hf_hub_download("raihan-js/oraclebench-items", "judge_runs/qwen3b_pointwise.jsonl", repo_type="dataset")
    p05 = hf_hub_download("raihan-js/oraclebench-items", "judge_runs/qwen05b_pointwise.jsonl", repo_type="dataset")
    r3 = [json.loads(l) for l in open(p3) if l.strip()]
    r05 = [json.loads(l) for l in open(p05) if l.strip()]
    assert len(r3) == len(r05) == 1760
    Y, F = [], []
    for a, b in zip(r3, r05):
        if a["oracle_verdict"] is False:  # truly-wrong subset (n=1333, incl. unparsed)
            Y.append(1.0 if a["judge_verdict"] == "CORRECT" else 0.0)
            F.append(1.0 if b["judge_verdict"] == "CORRECT" else 0.0)
    return np.array(Y), np.array(F)


def load_pairwise() -> tuple[np.ndarray, np.ndarray]:
    p3 = hf_hub_download("raihan-js/oraclebench-items", "judge_runs/qwen3b_pairwise_matched.jsonl", repo_type="dataset")
    p05 = hf_hub_download("raihan-js/oraclebench-items", "judge_runs/qwen05b_pairwise_matched.jsonl", repo_type="dataset")
    r3 = [json.loads(l) for l in open(p3) if l.strip()]
    r05 = [json.loads(l) for l in open(p05) if l.strip()]
    Y, F = [], []
    for a, b in zip(r3, r05):
        if a.get("order") not in ("AB", "BA") or a["picked_wrong"] is None:
            continue
        Y.append(0.0 if a["picked_wrong"] else 1.0)
        F.append(0.0 if b["picked_wrong"] else (1.0 if b["picked_wrong"] is False else 0.5))
    return np.array(Y), np.array(F)


def reestimate(name: str, Y: np.ndarray, F: np.ndarray, published: float, n: int = 100, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Y), n, replace=False)
    y = list(Y[idx])
    out = {"published": published, "full": float(Y.mean()), "n": n, "N": len(Y)}
    r = human_only(y)
    out["E1"] = {"est": r.value, "lo": r.lo, "hi": r.hi, "covers_full": bool(r.lo <= Y.mean() <= r.hi)}
    r = ppi_mean(list(F), y, list(F[idx]))
    out["E2"] = {"est": r.value, "lo": r.lo, "hi": r.hi, "covers_full": bool(r.lo <= Y.mean() <= r.hi)}
    mask = np.zeros(len(Y), bool)
    mask[idx] = True
    r = ppi_plus_mean(list(F[~mask]), y, list(F[idx]))
    out["E3"] = {"est": r.value, "lo": r.lo, "hi": r.hi, "covers_full": bool(r.lo <= Y.mean() <= r.hi)}
    return {name: out}


def main() -> None:
    res = {}
    Y, F = load_pointwise()
    assert len(Y) == 1333 and abs(Y.mean() - 147 / 1333) < 1e-9, (len(Y), Y.mean())
    res.update(reestimate("pointwise-false-accept", Y, F, 147 / 1333))
    Y2, F2 = load_pairwise()
    assert len(Y2) == 471 and abs(Y2.mean() - 379 / 471) < 1e-9, (len(Y2), Y2.mean())
    res.update(reestimate("pairwise-picks-right", Y2, F2, 379 / 471))
    with open("results/dogfood.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
