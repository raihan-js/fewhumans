# FewHumans JudgeBank — datasheet

One wide parquet per task: `item_id, prompt, human, truth_raw` plus per-judge
`{judge}__label/confidence/tokens_in/tokens_out/cost_usd/latency_s/parse_status/prompt_hash/model`.
Judges: `local-qwen35-9b` (Ollama Q4_K_M, think off), `local-qwen35-4b`,
`api-fast` (gpt-4o-mini, temp 0, verified 2026-10-10). Prompts frozen v1
(hashes in the files; templates in `study/run_judges/run_local.py`).
Total API spend for the whole bank: $0.29.

| task | rows | human labels | source | licence | text included? |
|---|---|---|---|---|---|
| helpsteer2-correctness | 800 | HelpSteer2 validation `correctness` | nvidia/HelpSteer2 | CC-BY-4.0 | yes (attribution here) |
| summeval-consistency | 100 | mteb/summeval expert-mean `consistency` | mteb/summeval | MIT | yes |
| mtbench-winrate | 318 | lmsys/mt_bench_human_judgments `human` | lmsys/mt_bench_human_judgments | CC-BY-4.0 | yes (attribution here) |
| arena-winrate | 800 | lmarena-ai/arena-human-preference-55k slice (seed 0) | lmarena-ai/arena-human-preference-55k | Apache-2.0 | yes |
| wmt-mqm | 500 | RicardoRei mirror of WMT MQM (Online-W, en-de) | research use; original google/wmt-mqm-human-evaluation | yes (newstest research standard) |
| toxicity-flag | 800 | google/civil_comments slice (balanced) | google/civil_comments | CC0 | yes |
| oraclebench-truth | 800 | raihan-js/oraclebench-items (oracle verdicts) | owner's own | yes |
| fedproc-roles | 800 | audit/model-consensus labels — **WEAK TRUTH, not human** | owner's own | yes (flagged) |

Intended use: re-running prediction-powered inference simulations (see the
FewHumans repo `study/`). `human` is "what full human labeling concludes"
except fedproc-roles (model consensus — do not use as human truth).
Single rater per item everywhere (no inter-rater data). Not for training
judges to imitate (contamination of future evals); eval use only.
