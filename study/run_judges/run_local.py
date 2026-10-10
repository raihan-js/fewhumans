"""Run local (Ollama, $0) judges for one study task -> data/judgebank/{task}.parquet.

Usage:
  .venv/bin/python study/run_judges/run_local.py --task arena-winrate \\
      --judge local-qwen35-9b --n 800 --out data/judgebank/arena-winrate-9b.parquet

Row schema (frozen): item_id, human, truth_raw, judge, confidence, raw,
tokens_in/out, cost_usd, latency_s, parse_status, judge_name, model, prompt_hash.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tasks import LOADERS

from fewhumans.judges.runner import JudgeConfig, PriceTable, Runner

TMPL = {
    "ordinal": (
        "Rate the following on a {lo}..{hi} scale for {dimension}.\n{item}\n"
        'Reply with EXACTLY this JSON and nothing else: {{"label": <integer>, "confidence": <0..1>}}'
    ),
    "pairwise": (
        "Which response is better? Reply A, B, or tie.\n{item}\n"
        'Reply with EXACTLY this JSON and nothing else: {{"label": <"A"|"B"|"tie">, "confidence": <0..1>}}'
    ),
    "binary": (
        "Classify the following as {neg} or {pos}.\n{item}\n"
        'Reply with EXACTLY this JSON and nothing else: {{"label": <"LABEL">, "confidence": <0..1>}}'
    ),
    "categorical": (
        "Assign exactly one of these labels: {label_set}.\n{item}\n"
        "Copy the label EXACTLY, character for character, do not shorten or paraphrase.\n"
        'Reply with EXACTLY this JSON and nothing else: {{"label": <"LABEL">, "confidence": <0..1>}}'
    ),
    "real": (
        "Rate the quality of the translation from 0 (worst) to 100 (best).\n{item}\n"
        'Reply with EXACTLY this JSON and nothing else: {{"label": <integer>, "confidence": <0..1>}}'
    ),
}


def build_prompt(row: dict[str, Any]) -> tuple[str, str]:
    meta = row["meta"]
    kind = meta["kind"]
    item = str(row["prompt"])
    if kind == "ordinal":
        t = TMPL[kind].format(lo=meta["lo"], hi=meta["hi"], dimension=meta["dimension"], item=item)
    elif kind == "binary":
        neg, pos = meta["labels"]
        t = TMPL[kind].format(neg=neg, pos=pos, item=item)
    elif kind == "categorical":
        t = TMPL[kind].format(label_set=", ".join(meta["labels"]), item=item)
    else:
        t = TMPL[kind].format(item=item)
    digest = hashlib.sha256(TMPL[kind].encode()).hexdigest()[:8]
    return t, f"{kind}-v1-{digest}"


JUDGES = {
    "local-qwen35-9b": JudgeConfig(name="local-qwen35-9b", provider="ollama",
                                   model="qwen3.5:9b", base_url="http://localhost:11434/v1"),
    "local-qwen35-4b": JudgeConfig(name="local-qwen35-4b", provider="ollama",
                                   model="qwen3.5:4b", base_url="http://localhost:11434/v1"),
}


async def main_async(args: argparse.Namespace) -> None:
    rows = LOADERS[args.task](n=args.n, seed=args.seed)
    cfg = JUDGES[args.judge]
    runner = Runner(PriceTable(), cache_dir="data/cache/judges", concurrency=4)
    prompts = [build_prompt(r) for r in rows]
    calls = await runner.run_all(cfg, [(r["item_id"], p) for r, (p, _) in zip(rows, prompts)], max_cost_usd=args.max_cost)
    out_rows = []
    for r, c, (_, ph) in zip(rows, calls, prompts):
        out_rows.append({
            "item_id": r["item_id"], "human": r["human"], "truth_raw": str(r["truth_raw"]),
            "judge": c.label, "confidence": c.confidence, "raw": c.raw,
            "tokens_in": c.tokens_in, "tokens_out": c.tokens_out, "cost_usd": c.cost_usd,
            "latency_s": c.latency_s, "parse_status": c.parse_status,
            "judge_name": cfg.name, "model": cfg.model, "prompt_hash": ph,
        })
    df = pd.DataFrame(out_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(json.dumps({
        "task": args.task, "judge": args.judge, "n": len(df),
        "abstain_rate": float((df["parse_status"] != "ok").mean()),
        "spent_usd": runner.spent_usd, "out": str(out),
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="helpsteer2-correctness")
    ap.add_argument("--judge", default="local-qwen35-9b")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cost", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = f"data/judgebank/{args.task}-{args.judge.split('-')[-1]}.parquet"
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
