#!/bin/bash
# A2 API fast judge (gpt-4o-mini, $3/task rail, ~$2 total est). Log: runs-api.log
cd "$(dirname "$0")/../.."
V=.venv/bin/python
for task in helpsteer2-correctness summeval-consistency mtbench-winrate arena-winrate wmt-mqm toxicity-flag oraclebench-truth fedproc-roles; do
  echo "=== $(date -Is) $task x api-fast ==="
  $V study/run_judges/run_api.py --task $task --n 800 --max-cost 3.0 2>&1 | tail -8
done
echo "=== ALL DONE $(date -Is) ==="
