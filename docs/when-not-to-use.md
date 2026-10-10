# When NOT to use FewHumans

- **Items not fixed in advance.** The pool must be fixed; sampling
  probabilities are defined against it. Streaming settings need the
  sequential design (Zrnic & Candès §6), which v1 does not implement.
- **Humans themselves unreliable.** Truth = "what full human labeling would
  conclude". If inter-rater agreement is near chance, no statistics can save
  the estimand — fix the guidelines first.
- **Distribution shift between pool and claim.** The interval covers the
  pool mean. Claiming "our system is 81% accurate" on new data needs the new
  data in the pool.
- **Hand-picked human labels.** Estimation requires known probabilities;
  the library refuses hand-picked labels (descriptive agreement only).
- **High-stakes domains** (medical, legal): reduces human review, never
  removes it. Say so in every report.
- **n below validated ranges** (E4 <100; skewed tasks <100–200): the gate
  measured undercoverage there. The study grid respects these ranges.
