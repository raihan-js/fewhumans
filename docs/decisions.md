# FewHumans — decisions

 append-only, newest first.

## 2026-10-10 — Phase 7 dogfood (results/dogfood.json)
- OracleBench pointwise false-accept: published 147/1333 = 11.03% reproduced
  exactly from raw files; FewHumans n=100: E1 13.0% [6.3,19.7], E2 12.6%,
  E3 13.0% — all cover. No README correction needed.
- OracleBench pairwise picks-right: published 379/471 = 80.47% reproduced;
  n=100: E1 71.0% [62.0,80.0] MISSES on this seed, E2/E3 cover — PPI saved
  the draw (illustration, seed 7, not a claim).
- FedProc-Ledger deferred: no per-item judge verdicts published; gold is
  model-consensus. Queued for the VETR reviewer-gold round (needs joinable
  IDs + human gold).- Analysis code frozen at `2b8090e`; preregistration frozen at `2ee2c93`
  (hash self-recorded in follow-up `8f5a524`). Targets w, grid, censoring rule,
  hypotheses H1–H5, cost assumptions ($0.30/label [$0.10, $1.00]) all frozen.
- Adaptation logged: n_ref = min(400, N//2) for small pools (summeval N=100,
  mtbench N=318); E5+R3-tox100 kept with its known caveat; Alt-Test skipped
  (no multi-rater tasks); E0 grid runs despite determinism (uniform pipeline).

## 2026-10-10 — post-prereg library fix (logged, conservative)
- E3/E4 at n=N (census, only summeval-n100): estimators raised instead of
  returning the human-only mean. Fixed to reduce to human-only (exact for the
  stratum mean; conservative variance). Direction: makes E3/E4 look MORE like
  E1, i.e. against savings claims, so it cannot inflate H1/H3. Analysis code
  unchanged (frozen); 4 affected study configs re-run.

## 2026-10-10 — A1 local runs launched (approvals A1/A2-fast-only/A3 as-is)
- Owner approvals: A1 full local ($0), A2 fast API judge only (cap $10, key
  handoff pending), A3 task/panel list as planned. Strong-model subset dropped.
- Deps added to venv (free, no spend; study code needs them): `datasets`,
  `huggingface_hub` (loaders), `ppi_py` from git (cross-check only).
- Transport finding: Ollama's OpenAI-compatible endpoint ignores think=false —
  qwen3.5 fills context with thinking and returns EMPTY content (100% ABSTAIN
  that way). Runner gained a native `ollama` provider (/api/chat, think=false).
  Verified live: 50-item pilot 0% abstain, MAE 0.68 (0-4), p50 3.1s/item.
- Pilot quality notes: verbalized confidence 0.9-1.0 (overconfident; Phase 5
  calibration analysis will show it); categorical labels get SHORTENED by the
  model ('INCORPORATED' for 'INCORPORATED_BY_REFERENCE') — analysis must
  canonicalize (exact→prefix), prompt says EXACTLY; judge leans high (31/50 = 4).
- Prompt freeze v1 (hashed, never tuned against humans): ordinal v1-846b8229,
  pairwise v1-d920d6ef, binary v1-94371404, categorical v1-6b681adb,
  real v1-00b844bf.
- Task pins: arena subsample 800/55k (seed 0); toxicity = google/civil_comments
  CC0, balanced ~50/50; wmt-mqm = Online-W en-de only system in mirror (2,371
  segs → 500); mtbench alpaca-vs-gpt3.5 pair = 318 rows; fedproc gold =
  local labeled_mentions.parquet (2,663 audit labels) + HF mentions.parquet
  context; oracle = HF items.jsonl (oracle_verdict truth).
- Panel deviation (logged in judges.yaml): on-disk qwen3.5:9b/4b instead of
  Qwen2.5-7B/Llama-3.1-8B (no download, known-good serving).
- A1 running in background: study/run_judges/run_all_local.sh (8 tasks × 2
  judges, ~10k judgments, log study/run_judges/runs.log).

## 2026-10-10 — Phase 4 gate (R=2,000; report results/coverage_report.md)
- Round 1 (z-intervals): 15 fails. Fix: Student-t everywhere (df=n−1;
  E4: n−strata) — cleared 5 borderline n=50s. E3 still ppi_py-exact (point+λ;
  width differs by analytic t/z, tested).
- Round 2: 10 fails. E4+R1 n=50 excluded from defaults (needs n≥100);
  wmt-skew ranges (E1/E2 n≥100, E3/E4 n≥200, E5 all-n); E5+R3 tox-n100
  CONFIRMED at R=8,000 (0.9378±0.0027, IPW heavy tails) — E5+R3 stays in the
  study (H2 needs it) but the default E5 routing is R0/R5, with the corner
  listed in the report, not hidden.
- E6-seq covers 100% everywhere but stops late (conservative union bound).
- F2 (peeking): 0–1.3pp inflation at these (late-stopping) targets.
- Stress S1/S2/S3 all pass (PPI never uses judge confidence).

## 2026-10-10 — A2 launched (key in project .env, verified)
- Owner put OPENAI_API_KEY in fewhumans/.env (gitignored, spec-compliant).
  Empty ~/.config stub removed. run_api.py reads .env first, key-file fallback.
- Verification call: gpt-4o-mini, strict JSON ok, $0.000016. Model ID confirmed.
- Prices VERIFIED at platform.openai.com/docs/pricing: gpt-4o-mini $0.15/$0.60
  per 1M in/out (matches prices.yaml). Spend est ~$1.50-3 for 6,400 judgments;
  per-task rail $3, owner cap $10. A2 running (log runs-api.log), concurrent
  with A1 (network vs GPU, no contention).

## 2026-10-10 — fedproc-roles DROPPED from main study (7 tasks remain)
- Two independent failures: (1) the "gold" is model-consensus (coding agent +
  OpenAI judges; explicitly no human experts), so judge-vs-consensus agreement
  cannot stand in for judge-vs-human; the VETR reviewer gold has not landed.
  (2) Context join unreliable: gold rows carry no clause number or mention
  text, doc-only matching shows judges the wrong mention (agreement 3-8%,
  below the 1/9 chance rate — a data bug, not a judge finding).
- Parquets kept as weak-truth secondary data; $0.034 API spend sunk. Owner can
  reinstate if reviewer gold lands with joinable IDs.
- wmt-mqm scale note (kept): judges rate 0-100, truth is negative MQM. E3/E4/E5
  are affine-invariant in the predictions (λ absorbs scale/shift), so no
  re-run; E2 on wmt is scale-sensitive and will be flagged in analysis.

## 2026-10-10 — Phase 2 (estimators + routing)
- E3 mirrors `ppi_py` (`ppi.ppi_mean_pointestimate`/`ppi_mean_ci`, installed from
  git for cross-check only) to machine precision (worst diff 2.2e-16 over 20
  random configs; tests/test_pp_plus.py). Subtlety found by the cross-check:
  ppi_py's λ uses the *centered* covariance (gradients re-centered at their own
  means); first version used uncentered and was off by up to 6.6e-3.
- E2 uses the all-N prediction mean + unbiased (ddof=1) variances (textbook);
  E3 mirrors ppi_py (unlabeled-only mean, ddof=0). Difference is O(n/N);
  documented in both docstrings, not "fixed".
- E5 implements Zrnic & Candès 2024 §4 eq.(1)/§5 Prop.1, verified against the
  arXiv HTML full text. Adaptation: our pool uses fixed-size (R0/R1) or Poisson
  (R2–R5) batches instead of i.i.d. Bernoulli draws; the estimator is unchanged
  and unbiased (E[ξ_i]=π_i); the plug-in variance S²_δ/N ignores
  without-replacement covariances (conservative there). Uniform-π E5 == E2 point
  (tested); Monte-Carlo unbiasedness smoke-tested.
- R2–R5 use Poisson sampling (the paper's own ξ_i∼Bern(π) model) so inclusion
  probabilities are EXACT — successive/without-replacement weighting would make
  logged π_i approximate and break E5. queue() returns the realized set; target
  n is the expectation. Floor π_i ≥ 0.1·budget/N enforced (§6.2).
- E6 ships only the human-only union-bound Hoeffding CS (validity sketched in
  the docstring from Hoeffding + union bound; cites Howard et al. 2021 for the
  concept only). PPI++/active sequential stay behind experimental=True.
- R4 is a 3-feature numpy logistic (no sklearn dependency); R5 ε-adaptation is
  a documented heuristic "inspired by" Sfyraki & Wang FTRL — no regret claimed.
- `ppi` extra added to pyproject (git URL; listed stack dep, local install only
  so far — no spend, no publish).
- Demo (examples/phase2_synthetic.py → results/): judge-only 0.723±0.012 misses
  truth 0.645 (F4 preview); E3 ≤ E2 < E1 widths; E4+R1 best; E5+R3 WORSE than
  uniform at all budgets despite disagreement AUROC 0.735 for "judge wrong"
  (IPW variance dominates) — honest open H2 evidence, consistent with Sfyraki &
  Wang. 20 reps only; not a claim, a checkpoint illustration.

## 2026-10-10 — Phase 0 scaffold
- Repo lives at `research-learn/fewhumans/` (workspace convention; spec said `~/code/fewhumans`).
- `ppi_py` NOT pinned yet: PyPI package name unverified. Added as empty `ppi` extra;
  Phase 2 will verify `ppi_py` vs `ppi-py` and pin before any cross-check.
- Session persistence is JSONL+JSON for Phase 0/1 (human-readable, diffable).
  Parquet (`panel.parquet`) arrives with the judgebank writer in Phase 3.
- Name checks (2026-10-10, via web): PyPI `fewhumans` → 404 (free);
  GitHub `raihan-js/fewhumans` → 404 (free); HF dataset
  `raihan-js/fewhumans-judgebank` → 401 on public fetch (inconclusive, but
  namespaced — owner can create). Fallback PyPI name reserved: `fewhumans-eval`.
  Re-check at upload time.
- GPU: RTX 3060 12GB present (build machine).
