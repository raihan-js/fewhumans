# FewHumans — pre-registration (stub, Phase 0)

Committed before running Phase 5. The primary metric and hypotheses below are the
spec (fewhumans.md §9.1); exact target half-widths per task and the analysis code
hash will be filled in at Phase 5 and this file re-committed BEFORE the main study.

- **Primary metric:** human-label savings at equal precision. For a target half-width
  w per task (w = half-width a 400-label human-only sample achieves), the number of
  human labels each method needs to reach w, relative to human-only (E1+R0), as a
  ratio with a bootstrap CI over replicates, pooled across tasks (random-effects).
- **Secondary:** coverage; mean half-width at fixed budgets; judge-only error and its
  coverage; peeking inflation; dollars per valid estimate (judge + human $/label,
  sensitivity range).
- **Hypotheses (open):** H1 PPI++ (E3+R0) beats human-only on most tasks. H2
  disagreement routing (R3) beats uniform (R0) under E5 (may fail per Sfyraki & Wang).
  H3 stratification (E4+R1) captures most of active routing's benefit. H4 judge-only
  (E0) misses truth outside its own interval on most tasks. H5 panel beats best
  single judge.
- **Commit hash of this registration:** TBD at Phase 5.
