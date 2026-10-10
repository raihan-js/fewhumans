"""Run the approved API fast judge ($10 cap) for one study task.

Usage:
  .venv/bin/python study/run_judges/run_api.py --task arena-winrate --n 800

Key: read at runtime from ~/.config/fewhumans/openai_key (chmod 600).
Never logged, never committed, never passed anywhere except the HTTPS header.
Same frozen prompts (study/run_judges/run_local.py TMPL) and row schema.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_local import JUDGES, main_async

from fewhumans.judges.runner import JudgeConfig, PriceTable

KEY_FILE = Path.home() / ".config" / "fewhumans" / "openai_key"
DOTENV = Path(__file__).resolve().parents[2] / ".env"
# Prices VERIFIED 2026-10-10 against https://platform.openai.com/docs/pricing
# (Standard tier, per 1M tokens). Re-verify if a run happens much later.
PRICES = PriceTable(
    per_1k_in={"gpt-4o-mini": 0.00015, "gpt-4.1-mini": 0.00040},
    per_1k_out={"gpt-4o-mini": 0.00060, "gpt-4.1-mini": 0.00160},
)


def read_key() -> str:
    for source in (DOTENV, KEY_FILE):
        if source.exists():
            for line in source.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY"):
                    key = line.split("=", 1)[1].strip().strip("\"").strip("'")
                    if key:
                        return key
    raise SystemExit(
        f"no OPENAI_API_KEY in {DOTENV} or {KEY_FILE}. Store it with:\n"
        '  printf %s "sk-..." >> .env  (project folder, gitignored)'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--judge-name", default="api-fast")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cost", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    key = read_key()
    JUDGES[args.judge_name] = JudgeConfig(
        name=args.judge_name, provider="openai_compatible", model=args.model,
        base_url="https://api.openai.com/v1", api_key=key,
    )
    args.judge = args.judge_name
    args.prices = PRICES
    if args.out is None:
        args.out = f"data/judgebank/{args.task}-apifast.parquet"
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
