# fewhumans — project rules

## What this is
A library that combines a few human labels with many LLM-judge labels to give unbiased
estimates with valid (including anytime-valid) confidence intervals, decides which items
humans should label, and tells researchers when to stop. Plus an empirical study of which
strategies work on real annotation tasks. Plan: docs/PLAN.md. Decisions: docs/decisions.md.
Pre-registration: docs/preregistration.md.

## Non-negotiables
- No estimator/interval reaches the public API until it passes the Phase 4 coverage
  simulations (empirical coverage within Monte Carlo error of the nominal level).
- Human labels used for estimation must be sampled with known, logged probabilities.
  The library refuses to estimate from hand-picked human labels (it can still compute
  descriptive agreement on them, clearly labeled "not an estimate").
- No invented numbers or citations. Reported numbers come from results/*.json.
- Ground-truth human labels in study datasets are hidden from strategies during
  simulation; only the simulator "reveals" a label when a strategy buys it.
- Secrets in .env only. Never commit data/.
- Ask before: >$5 spend per run, new dependencies, publishing anywhere,
  changing the pre-registered primary metric.

## Stack
Python 3.10+, uv, numpy, scipy, pandas, pyarrow, pydantic v2, typer, rich, httpx,
tenacity, ppi_py (reference + cross-check), matplotlib, pytest, hypothesis,
ruff, mypy (strict on src/fewhumans), mkdocs-material.
Judges: openai-compatible client (works with vLLM, Ollama and API providers),
anthropic SDK. Local judges via vLLM (Linux/WSL2) or Ollama.
Optional UI: streamlit.

## Layout
src/fewhumans/{core,judges,routing,estimators,sequential,report,io,cli,ui}
study/{datasets,run_judges,simulate,analyze,report}
tests/  docs/  paper/  results/ (small JSON, committed)  data/ (gitignored)

## Commands
uv sync · uv run pytest -q · uv run ruff check . · uv run mypy src
uv run fewhumans --help · uv run fh-study --help

## Style
Pure, typed functions for all statistics, each with a docstring stating the estimand,
assumptions and the paper it implements (with equation numbers). Property-based tests
(hypothesis) for estimators. Every decision logged in docs/decisions.md.
