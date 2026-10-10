# Examples

Runnable with `pip install fewhumans` (run from a repo checkout).
Each is smoke-tested in CI (`tests/test_examples.py`).

- **`examples/quickstart_accuracy.py`** — the canonical flow. An LLM judge
  labeled 1,000 responses; you label 50. Compares judge-only (biased) vs
  human-only vs PPI++, checks the stopping rule, writes the report card.
  Start here.
- **`examples/pairwise_winrate.py`** — which system wins? Pairwise A/B/tie
  judgments, disagreement routing (R3), E1 vs E5 active estimation.
- **`examples/toy_end_to_end.py`** — minimal Session round trip (fake judge).
- **`examples/phase2_synthetic.py`** — reproduces the Checkpoint-2 demo
  figure (`results/phase2_synthetic.png`).

When to use what (from the pre-registered study): uniform humans + E3/E4 by
default; E5 for heavy-tailed scores; never report E0; skip active routing
unless a pilot shows concentrated, predictable judge errors. Full table:
README "Which estimator when".
