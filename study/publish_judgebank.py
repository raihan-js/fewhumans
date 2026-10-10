"""Build the JudgeBank release (D3): one wide parquet per task + datasheet.

Wide schema: item_id, prompt, human, truth_raw,
  {judge}__label, {judge}__confidence, {judge}__tokens_in/out, {judge}__cost_usd,
  {judge}__latency_s, {judge}__parse_status, {judge}__prompt_hash.
Source text included only where the licence allows (see DATASHEET.md);
otherwise item IDs + join script (this file) reproduce the join.
fedproc-roles ships as weak-truth (model-consensus gold), clearly flagged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "run_judges"))
from tasks import LOADERS

JUDGES = {"9b": "local-qwen35-9b", "4b": "local-qwen35-4b", "apifast": "api-fast"}
TASKS = {
    "helpsteer2-correctness": {"n": 800, "licence": "CC-BY-4.0", "text": True},
    "summeval-consistency": {"n": 100, "licence": "MIT", "text": True},
    "mtbench-winrate": {"n": 318, "licence": "CC-BY-4.0", "text": True},
    "arena-winrate": {"n": 800, "licence": "Apache-2.0", "text": True},
    "wmt-mqm": {"n": 500, "licence": "newstest research use (see datasheet)", "text": True},
    "toxicity-flag": {"n": 800, "licence": "CC0", "text": True},
    "oraclebench-truth": {"n": 800, "licence": "owner's own (raihan-js)", "text": True},
    "fedproc-roles": {"n": 800, "licence": "owner's own; WEAK-TRUTH (model consensus)", "text": True},
}


def main() -> None:
    out_dir = Path("data/release/judgebank")
    out_dir.mkdir(parents=True, exist_ok=True)
    for task, spec in TASKS.items():
        rows = LOADERS[task](n=spec["n"], seed=0)
        base = pd.DataFrame([{"item_id": r["item_id"], "prompt": r["prompt"],
                              "human": r["human"], "truth_raw": str(r["truth_raw"])} for r in rows])
        for suffix, jname in JUDGES.items():
            p = Path(f"data/judgebank/{task}-{suffix}.parquet")
            if not p.exists():
                print(f"  SKIP {task}/{jname} (missing)")
                continue
            df = pd.read_parquet(p).set_index("item_id")
            for col in ["judge", "confidence", "tokens_in", "tokens_out", "cost_usd",
                        "latency_s", "parse_status", "prompt_hash", "model"]:
                base[f"{jname}__{col}"] = base["item_id"].map(df[col])
        base.to_parquet(out_dir / f"{task}.parquet", index=False)
        print(f"  {task}: {len(base)} rows -> {out_dir / f'{task}.parquet'}")
    print("done.")


if __name__ == "__main__":
    main()
