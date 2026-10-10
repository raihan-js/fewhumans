# Phase 4 coverage gate report (2026-10-10, R=2,000; marginal rerun R=8,000)

Pass rule: empirical coverage ≥ 0.95 − 2·MCSE. MCSE ≈ 0.005 at R=2,000.

## Main gate: 196 configs (7 tasks × 7 methods × budgets 50/100/200/400)
10 failures. Full table: results/coverage.json. Failures: results/coverage_failures.json.

| failure | coverage | cause |
|---|---|---|
| E4+R1 n=50 (arena, helpsteer, oracle) | 0.936–0.939 | per-stratum n≈12: CLT too weak |
| E5+R3 toxicity n=100 | 0.936 (R=2k) → **0.9378±0.0027 (R=8k, confirmed)** | IPW heavy tails (small π, rectifier ±1) |
| wmt E1 n=50 / E2 n=50 | 0.913 / 0.937 | heavy-tailed MQM truth, n too small |
| wmt E3 n=50/n=100, E4 n=50/n=100 | 0.883–0.933 | λ-estimation noise + skew |

Fixed by construction change: all CLT intervals now use Student-t
(df = n−1 rectifier; E4: n−strata) — this cleared 5 borderline n=50 fails
(E1/E3/E2 on non-skewed tasks). Point estimates and λ unchanged (E3 still
matches ppi_py exactly up to the analytic t/z width factor).

## Validated default ranges (the gate's verdict)
- E1+R0, E2+R0: n ≥ 50 all tasks; wmt-like skew: n ≥ 100.
- E3+R0: n ≥ 50; wmt-like skew: n ≥ 200.
- E4+R1: n ≥ 100; wmt-like skew: n ≥ 200.
- E5+R0, E5+R5: all n, all tasks. **E5+R3: default except the
  toxicity-n≈100 corner (true cov ≈ 0.938, ~1pp under)** — listed, not hidden;
  the Phase 5 study still evaluates E5+R3 (that is hypothesis H2).
- E6-seq (union-bound CS): 100% coverage everywhere, but conservative —
  often stops at the cap. Valid; stopping savings reported in sequential.json.

## Sequential (stopping at E1-400 half-width), R=2,000
E6 coverage 1.000 on all 7 tasks (mean stop n: at cap for small tasks).
Full: results/sequential.json.

## Peeking = Finding F2 (fixed E2 interval re-checked each label)
Coverage at stop: 1.000 ×5, toxicity 0.993, oracle 0.987. Stopping was late
(stop n 441–800), so selection bias cost only ~0–1.3pp here. Honest scope:
with aggressive early targets the cost would be larger (not simulated).

## Stress (synthetic, n=100, R=2,000): results/stress.json
S1 (99%/30% hard): 0.951–0.964. S2 (class-biased): 0.943–0.968.
S3 (miscalibrated conf): 0.948–0.958. All pass — PPI needs no calibrated
confidence (it never uses it).
