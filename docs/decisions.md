# FewHumans — decisions

 append-only, newest first.

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
