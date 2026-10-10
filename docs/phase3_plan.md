# FewHumans Phase 3 plan (2026-10-10) — NEEDS OWNER APPROVAL BEFORE ANY RUN

## Recon (done, $0)
- GPU: RTX 3060 12GB. Disk: 120GB free (watch it; judgebank will add ~1-5GB).
- On-disk judges (no download): Ollama `qwen3.5:9b` (6.6GB, served before),
  `qwen3.5:4b` (3.4GB). No Qwen2.5-7B/Llama-3.1-8B (deviation from spec, logged
  in judges.yaml; downloads are free but unnecessary).
- vLLM not installed; Ollama OpenAI-compatible API is the local transport
  (logprob support VERIFY-AT-BUILD; verbalized + self-consistency definite).
- Keys: no ANTHROPIC/OPENAI env keys; owner's OpenAI key exists (fedproc-ledger
  used OpenAI judges) — reuse needs owner confirmation, never copied to repo.
- Datasets verified: HelpSteer2 (CC-BY-4.0), MT-Bench human (CC-BY-4.0),
  Arena-55k (Apache-2.0), SummEval (mteb/summeval, licence TBD),
  WMT-MQM (original GitHub, licence TBD). Toxicity/HANNA/NER VERIFY-AT-BUILD.

## Steps (in order)
1. Local pilot ($0): runner → Ollama `qwen3.5:9b`, 50 items × HelpSteer2.
   Validates transport/prompts/parsing/cache. No approval needed (no spend).
2. Freeze prompts (hash → docs/decisions.md), download task slices (free).
3. Local full runs ($0): 8 tasks × ~800 items × 2 local judges (+ self-consistency
   k=5 on a 200-item subset). ~10k judgments, a few GPU-hours.
4. API fast runs (APPROVAL NEEDED): 8 × 800 × 1 judge ≈ 6,400 judgments.
5. API strong subset (SEPARATE APPROVAL): 2 tasks × 800 × 1 judge.
6. Checkpoint 3: per task×judge accuracy/agreement, abstain rate, cost, total.

## Spend estimate (prices ESTIMATE — re-verify at build, see prices.yaml)
| run | judgments | ≈tokens | est. cost |
|---|---|---|---|
| API fast (8×800) | 6,400 | ~10.6M | ~$2–4 |
| API strong (2×800) | 1,600 | ~2.6M | ~$8–12 |
| Total | | | **~$10–16** |
Proposed cap: **$25** (buffer for retries/self-consistency). Per-run cost cap
enforced in code (`max_cost_usd`), disk cache makes re-runs free.

## Approvals needed from owner
- [ ] A1: run local pilot + full local judges ($0, GPU time)
- [ ] A2: reuse OpenAI key + spend up to $25 (fast judge; strong subset included?)
- [ ] A3: task list (8 tasks) + panel (2 local + 1-2 API) as above
