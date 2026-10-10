# Concepts (plain language)

## The problem
You used an LLM as a judge on 5,000 items and report "accuracy 81%". Two
things are wrong: the judge's errors flow straight into the number (bias),
and the ±2% you computed describes the judge's opinions, not the truth.

## What FewHumans does
1. The judge labels everything (cheap).
2. A few items are sampled for humans **with known probabilities** (random,
   stratified, or active). Known probabilities are the whole game: they let
   us re-weight the human labels so the estimate stays unbiased.
3. Prediction-powered inference (PPI) combines both: the judge gives the
   rough shape, the human labels correct it. The interval covers the **true**
   value (what full human labeling would conclude) — proven by the coverage
   gate (results/coverage_report.md), not by assertion.
4. Anytime-valid sequences let you peek after every label and stop when the
   interval is narrow enough. Ordinary intervals break when peeked
   (we measured 0–1.3pp inflation: results/peeking.json).

## Why sampling probabilities matter
A hand-picked "interesting 100" cannot be used for estimation — the library
refuses (it will still show descriptive agreement, clearly labeled as such).
Uniform, stratified, and Poisson active sampling all produce exact,
logged probabilities.

## What the study found (7 tasks, pre-registered)
PPI++ and stratification saved ~9% of human labels; fixed PPI and
disagreement-routing IPW *cost* labels (2.3×/2.6×) with these mediocre
judges; judge-only intervals missed truth 18/21 times; verbalized confidence
is overconfident everywhere (ECE up to 0.98). Details: results/study_report.md.
