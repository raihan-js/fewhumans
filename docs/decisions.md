# FewHumans — decisions

 append-only, newest first.

## 2026-10-10 — Phase 0 scaffold
- Repo lives at `research-learn/fewhumans/` (workspace convention; spec said `~/code/fewhumans`).
- `ppi_py` NOT pinned yet: PyPI package name unverified. Added as empty `ppi` extra;
  Phase 2 will verify `ppi_py` vs `ppi-py` and pin before any cross-check.
- Session persistence is JSONL+JSON for Phase 0/1 (human-readable, diffable).
  Parquet (`panel.parquet`) arrives with the judgebank writer in Phase 3.
- Name checks (2026-10-10, via web): PyPI `fewhumans` → 404 (free);
  GitHub `raihan-js/fewhumans` → 404 (free); HF dataset
  `raihan-js/fewhumans-judgebank` → 401 on public fetch (inconclusive, but
  namespaced — owner can create). Fallback PyPI name reserved: `fewhumans-eval`.
  Re-check at upload time.
- GPU: RTX 3060 12GB present (build machine).
