# FewHumans — pre-registration for the Phase 5 main study

Committed 2026-10-10 BEFORE the main study ran. Analysis code frozen at
commit `2b8090e82a6792656bb9cee52a6277ea69ae8488`
(study/simulate/study.py, study/analyze/study.py). Any later code change is
logged in docs/decisions.md. This file's commit hash is recorded there.

## Primary metric: human-label savings at equal precision
For target half-width w per task (frozen table below = analytic E1
half-width, Student-t, at n_ref; n_ref = min(400, N//2) — the min() adapts the
spec's n=400 rule to small pools: summeval N=100, mtbench N=318):

| task | N | n_ref | w |
|---|---|---|---|
| helpsteer2-correctness | 800 | 400 | 0.1217 |
| summeval-consistency | 100 | 50 | 0.4420 |
| mtbench-winrate | 318 | 159 | 0.0500 |
| arena-winrate | 800 | 400 | 0.0412 |
| wmt-mqm | 800 | 400 | 0.2947 |
| toxicity-flag | 800 | 400 | 0.0492 |
| oraclebench-truth | 795 | 397 | 0.0412 |

For each (task, method, panel): mean half-width curve over budget grid
{25,50,75,100,150,200,300,400,600} (∩ feasible ∩ gate ranges), R=2,000
replicates; labels-to-w by linear interpolation (curve enforced
non-increasing); savings ratio = n_method / n_E1+R0; 95% CI by nonparametric
bootstrap (B=1,000) over replicates. Censoring rule (frozen): ratio uses
grid-end ×2 when w is never reached (reported as-is, conservative against
the method). Pooled across tasks: DerSimonian-Laird random-effects on
log-ratios.

Gate ranges enforced (from results/coverage_report.md): E4+R1 n≥100;
wmt-mqm E1/E2 n≥100, E3/E4 n≥200. E5+R3 kept everywhere including the
toxicity-n≈100 corner (known ~1pp undercoverage, listed).

## Secondary
- Coverage at fixed budgets (from the same runs); E0 judge-only error and its
  naive-interval coverage (H4); peeking inflation (have: results/peeking.json);
  E6 stopping savings (have: results/sequential.json).
- Cost per valid estimate: judge $ (measured unit costs × N panel judgments;
  local $0, api-fast actuals) + human labels × ASSUMED $0.30/label, sensitivity
  $0.10/$1.00 (assumption, not a fact).
- Calibration: ECE + reliability bins of verbalized confidence per judge/task.
- Routing-help: relate E5+R3-vs-E5+R0 savings to AUROC(disagreement→wrong),
  panel agreement, truth spread, label space (7 tasks: descriptive, no
  p-values). Alt-Test skipped: no task has multiple human raters per item.

## Hypotheses (open, any outcome reported)
- H1: PPI++ (E3+R0/full) needs fewer labels than human-only on most tasks
  (pooled ratio < 1).
- H2: E5+R3 needs fewer labels than E5+R0. Prior (Sfyraki & Wang + our gate)
  says it may fail; reported either way.
- H3: E4+R1 captures most of E5+R3's benefit (ratio of savings).
- H4: E0 misses truth outside its own interval on a majority of
  task×panels (21 combos).
- H5: full panel gives larger savings than the best single judge (E2/E3:
  full vs best).

## Commit hash of this registration
(TBD: filled at commit time — see docs/decisions.md.)
