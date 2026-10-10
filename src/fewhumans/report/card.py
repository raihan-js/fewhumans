"""Minimal HTML judge report card (spec §10)."""

from __future__ import annotations

from typing import Any


def render_html(session: Any, estimate: Any = None) -> str:
    n, nh = len(session.items), len(session.humans)
    agr = session.agreement()
    est_line = str(estimate) if estimate else "no estimate yet"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>FewHumans report</title></head><body>
<h1>FewHumans report</h1>
<p>Items: {n}. Human labels: {nh}. Panel: {[j.name for j in session.panel]}.</p>
<h2>Estimate</h2><p>{est_line}</p>
<h2>Judge agreement (descriptive)</h2><p>{agr}</p>
<p><em>Phase-1 minimal card. Calibration plots, strata and disagreements land in Phase 6.</em></p>
</body></html>"""
