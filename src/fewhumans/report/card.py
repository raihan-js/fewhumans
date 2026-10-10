"""Judge report card (spec §10): history, strata, calibration, disagreements.

Everything shown is computed from the session's own data — no invented numbers.
Figures are matplotlib PNGs embedded as base64.
"""

from __future__ import annotations

import base64
import html
from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fig_to_img(fig: Any) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def history_plot(history: list[dict[str, Any]]) -> str:
    fig, ax = plt.subplots(figsize=(6, 3))
    ns = [h.get("n_human", 0) for h in history]
    vs = [h.get("value", float("nan")) for h in history]
    los = [h.get("lo", float("nan")) for h in history]
    his = [h.get("hi", float("nan")) for h in history]
    ax.errorbar(ns, vs, yerr=[np.array(vs) - np.array(los), np.array(his) - np.array(vs)],
                fmt="o-", capsize=3, markersize=3)
    ax.set_xlabel("human labels")
    ax.set_ylabel("estimate")
    ax.set_title("Estimate history (interval vs human labels)")
    return _fig_to_img(fig)


def calibration_rows(session: Any) -> list[dict[str, Any]]:
    """Correctness vs confidence bins from human-labeled items."""
    est = session.task.estimand
    rows = []
    for iid, h in session.humans.items():
        p = session._panel_pred(iid)
        if p is None:
            continue
        confs = list((session.panel_conf.get(iid) or {}).values())
        c = float(confs[0]) if confs and confs[0] is not None else None
        try:
            ok = abs(float(est.map(p)) - float(est.map(h))) < 1e-9
        except (TypeError, ValueError):
            ok = str(p) == str(h)
        rows.append({"conf": c, "ok": ok})
    return rows


def plain_summary(session: Any, res: Any) -> str:
    n = len(session.humans)
    base = (f"With {n} human labels, the estimate is {res.value:.3f}, between "
            f"{res.lo:.3f} and {res.hi:.3f} with {int(res.confidence*100)}% confidence "
            f"({res.method}).")
    try:
        e0 = session.estimate("E0-judge-only")
        base += f" Judge-only would have said {e0.value:.3f}."
    except (ValueError, KeyError):
        pass
    return base


def render_html(session: Any, estimate: Any = None) -> str:
    n, nh = len(session.items), len(session.humans)
    est_line = html.escape(str(estimate)) if estimate else "no estimate yet"
    parts = ["<h1>FewHumans report</h1>",
             f"<p>Items: {n}. Human labels: {nh}. Panel: {[j.name for j in session.panel]}.</p>",
             f"<h2>Estimate</h2><p>{est_line}</p>"]
    if estimate is not None:
        parts.append(f"<h2>Plain-language summary</h2><p>{html.escape(plain_summary(session, estimate))}</p>")
    if session.history:
        parts.append("<h2>Estimate history</h2>" + f'<img src="data:image/png;base64,{history_plot(session.history)}"/>')
        rows = "".join(
            f"<tr><td>{html.escape(str(h.get('method','')))}</td><td>{h.get('n_human')}</td>"
            f"<td>{h.get('value'):.4f}</td><td>[{h.get('lo'):.4f}, {h.get('hi'):.4f}]</td></tr>"
            for h in session.history[-20:])
        parts.append(f"<table border=1><tr><th>method</th><th>n</th><th>value</th><th>interval</th></tr>{rows}</table>")
    # Agreement by stratum (descriptive, not estimates).
    from fewhumans.estimators.stratified import panel_agreement_strata

    strata = panel_agreement_strata(session.panel_outputs)
    by_h: dict[str, list[int]] = {}
    for iid, h in session.humans.items():
        p = session._panel_pred(iid)
        if p is None:
            continue
        by_h.setdefault(strata.get(iid, "?"), []).append(1 if str(p) == str(h) else 0)
    if by_h:
        rows = "".join(f"<tr><td>{html.escape(h)}</td><td>{len(v)}</td><td>{np.mean(v):.3f}</td></tr>" for h, v in by_h.items())
        parts.append("<h2>Judge agreement by stratum (descriptive)</h2>"
                     f"<table border=1><tr><th>stratum</th><th>n</th><th>agreement</th></tr>{rows}</table>")
    # Calibration.
    cal = calibration_rows(session)
    known = [r for r in cal if r["conf"] is not None]
    if known:
        ece = float(np.mean([abs((1.0 if r["ok"] else 0.0) - float(r["conf"])) for r in known]))
        parts.append(f"<h2>Confidence calibration</h2><p>Mean absolute gap between "
                     f"confidence and correctness: {ece:.3f} (n={len(known)}). Lower is better; "
                     f"verbalized confidence is typically overconfident.</p>")
    # Most informative disagreements: confident panel mistakes first.
    dis = []
    for iid, h in session.humans.items():
        p = session._panel_pred(iid)
        if p is None or str(p) == str(h):
            continue
        confs = [c for c in (session.panel_conf.get(iid) or {}).values() if c is not None]
        dis.append((max([float(c) for c in confs] or [0.0]), iid, p, h))
    dis.sort(reverse=True)
    if dis:
        rows = "".join(
            f"<tr><td>{html.escape(iid)}</td><td>{c:.2f}</td><td>{html.escape(str(p))}</td>"
            f"<td>{html.escape(str(h))}</td><td>{html.escape(str(session.panel_outputs.get(iid, {})))}</td></tr>"
            for c, iid, p, h in dis[:20])
        parts.append("<h2>Most informative disagreements (top 20)</h2>"
                     "<table border=1><tr><th>item</th><th>confidence</th><th>panel</th><th>human</th><th>all judges</th></tr>"
                     f"{rows}</table>")
    parts.append("<p><em>Agreement and calibration sections are descriptive, not estimates. "
                 "Estimates use only human labels drawn with known probabilities.</em></p>")
    return "<!doctype html><html><head><meta charset='utf-8'><title>FewHumans report</title></head><body>" + "".join(parts) + "</body></html>"
