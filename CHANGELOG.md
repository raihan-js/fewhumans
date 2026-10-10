# Changelog

## 1.0.0 (2026-10-10)
- E0–E6 estimators (PPI++ matches ppi_py to 2.2e-16; Student-t intervals),
  R0–R5 routing, anytime-valid CS, Wilson proportions.
- Async judge runner (OpenAI-compatible/Anthropic/Ollama-native), disk cache,
  cost caps, strict-schema parsing with ABSTAIN.
- Session + CLI (`init/judge/queue/add/label/estimate/report`) + `fh-study`.
- Coverage gate (196 configs, R=2,000), pre-registered 7-task study
  (846 configs), dogfood re-estimates, report card, docs.
