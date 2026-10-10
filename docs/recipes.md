# Recipes per label space

Map judge/human labels to numbers with `Mean(of=...)`, then pick the method
the gate validated for your budget (see results/coverage_report.md ranges).

- **Binary** (toxicity, oracle truth): `Mean(of=lambda y: y == "correct")`.
  E1/E2/E3/E5 all validated n≥50. Wilson (`human_proportion_wilson`) for
  human-only proportions.
- **Pairwise** (win rate): map A→1, B→0, tie→0.5 for both judges and humans.
  Same methods as binary.
- **Ordinal** (1–5 ratings): map to float. E1/E2/E3 validated n≥50; E4 needs
  n≥100 (per-stratum n too small below that).
- **Real** (MQM scores): map to float. Heavy-tailed truth needs care —
  on wmt-mqm, E1/E2 needed n≥100 and E3/E4 n≥200; E5 held at all n.
  Judge and truth need not share a scale for E3/E4/E5 (λ absorbs affine
  transforms); E2 is scale-sensitive.
- **Categorical** (roles): estimate the prevalence of one class
  (`Mean(of=lambda y: y == "FULL_TEXT")`) or accuracy-vs-gold per class.
- **Ratio estimands** (precision/recall/F1): NOT shipped as means — the delta
  method and bootstrap variants did not go through the gate in v1.
  Documented, not implemented.
