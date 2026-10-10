#!/bin/bash
# A1 full local runs: 8 tasks x 2 judges. Logs to study/run_judges/runs.log
cd "$(dirname "$0")/../.."
V=.venv/bin/python
for judge in local-qwen35-9b local-qwen35-4b; do
  for task in helpsteer2-correctness summeval-consistency mtbench-winrate arena-winrate wmt-mqm toxicity-flag oraclebench-truth fedproc-roles; do
    echo "=== $(date -Is) $task x $judge ==="
    $V study/run_judges/run_local.py --task $task --judge $judge --n 800 2>&1 | tail -8
  done
done
echo "=== ALL DONE $(date -Is) ==="
