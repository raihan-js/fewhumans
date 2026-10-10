# Phase 5 main study — what the evidence supports (2026-10-10)

Pre-registered (docs/preregistration.md, frozen before runs). 7 tasks, R=2,000.
Primary metric: human labels to target half-width w vs E1+R0 (ratio + bootstrap
CI; DL random-effects pool). Full numbers: results/study_summary.json.
Figures: results/forest.png, results/calibration.png.

## Pooled savings ratios (DL random-effects, 7 tasks)
| method | ratio [95% CI] | reading |
|---|---|---|
| E3+R0/full (PPI++) | 0.908 [0.865, 0.952] | saves ~9% of labels |
| E4+R1/full (stratified) | 0.907 [0.843, 0.977] | saves ~9% |
| E5+R0/full (active est, uniform) | 1.239 [1.087, 1.412] | COSTS ~24% more |
| E5+R5/full (mixture) | 1.269 [1.055, 1.526] | costs ~27% more |
| E2+R0/full (fixed PPI) | 2.269 [1.558, 3.302] | costs 2.3×+ (censored lower bound) |
| E5+R3/full (disagreement) | 2.636 [2.505, 2.773] | costs 2.6×+ (censored lower bound) |

"Censored lower bound": the method never reached w within the budget grid on
most tasks (E2: 4/7, E5R3: 5/7); true ratios are ≥ reported. E5R3's implied n
exceeds pool N on 6/7 tasks — target unachievable there, not just expensive.

## Hypotheses
- H1 (E3 beats human-only): HOLDS weakly — pooled 0.91, per-task 0.84–0.99
  on all 7 tasks. ~9% savings, not ~50%.
- H2 (R3 beats R0 under E5): FAILS — 2.64 vs 1.24 pooled; R3 worse on 6/7
  tasks despite disagreement AUROC 0.61–0.80 for "judge wrong". IPW variance
  dominates the signal. (Consistent with Sfyraki & Wang and our gate.)
- H3 (stratification captures most of active benefit): MOOT — active routing
  has no benefit to capture; E4 ≈ E3 (≈9% savings, simpler assumptions).
- H4 (judge-only misses truth): HOLDS — E0 naive interval misses the
  full-human value in 18/21 task×panels.
- H5 (panel beats best single): FAILS — E3 full 0.908 vs best 0.918;
  negligible. (3 mediocre judges don't combine into a good one.)

## Secondaries
- Judge quality (Checkpoint 3): api-fast best/tied-best nearly everywhere,
  but absolute levels are mediocre (pairwise agreement 45–65%, ordinal
  corr 0.2–0.55). Garbage-in applies: weak judges → single-digit savings.
- Calibration: verbalized confidence is overconfident everywhere
  (ECE 0.17–0.98; wmt ~0.9). Never use it as a probability.
- Cost per valid estimate (human $0.30/label [$0.10, $1.00] assumed; judges
  measured: local $0, api-fast ≈$0.00009/judgment): E1 ≈ $120, E3 ≈ $100–116
  on N=800 tasks — PPI++ saves ~$10–20 per estimate. E5R3's target is
  unachievable within these pools.
- E6 stopping (sequential.json): valid (100% cover) but stops late.
- Peeking (peeking.json): 0–1.3pp inflation at late-stopping targets.

## Say / don't say
Say: "PPI++ and stratification saved ~9% of human labels; disagreement
routing cost 2.6×; judge-only intervals missed truth 18/21 times."
Never: "PPI halves labeling cost", "active learning wins", "the panel beats
single judges", or any claim about absolute judge quality beyond these tasks.
