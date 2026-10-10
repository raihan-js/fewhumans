"""Study task loaders: fixed slices with human ground truth (Phase 3).

Each loader returns rows: {item_id, prompt, human (float), truth_raw, meta}.
Deterministic (seeded); slices documented per task in study/configs/tasks.yaml.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def load_helpsteer2(n: int = 800, seed: int = 0) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("nvidia/HelpSteer2", split="validation")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), min(n, len(ds)), replace=False)
    rows = []
    for k in sorted(idx):
        r = ds[int(k)]
        rows.append({
            "item_id": f"helpsteer2-val-{k}",
            "prompt": f"Prompt: {r['prompt']}\nResponse: {r['response']}",
            "human": float(r["correctness"]),
            "truth_raw": int(r["correctness"]),
            "meta": {"kind": "ordinal", "lo": 0, "hi": 4, "dimension": "correctness"},
        })
    return rows


def load_summeval(system_idx: int = 0) -> list[dict[str, Any]]:
    """100 summaries from one summarizer; truth = expert-mean consistency."""
    from datasets import load_dataset

    ds = load_dataset("mteb/summeval", split="test")
    import json

    rows = []
    for doc in ds:
        machines = json.loads(doc["machine_summaries"]) if isinstance(doc["machine_summaries"], str) else list(doc["machine_summaries"])
        cons = json.loads(doc["consistency"]) if isinstance(doc["consistency"], str) else list(doc["consistency"])
        # Both align per summary (16 systems); take system_idx (expert mean).
        si = system_idx if system_idx < len(machines) else 0
        summ = machines[si]
        truth = float(cons[si]) if si < len(cons) else 3.0
        rows.append({
            "item_id": f"summeval-{doc['id']}-sys{system_idx}",
            "prompt": f"Article: {doc['text'][:2000]}\nSummary: {summ}",
            "human": truth,
            "truth_raw": truth,
            "meta": {"kind": "ordinal", "lo": 1, "hi": 5, "dimension": "consistency"},
        })
    return rows


def load_mtbench(pair: tuple[str, str] | None = None, max_n: int = 600) -> list[dict[str, Any]]:
    """Pairwise rows for one model pair; truth: A=1, B=0, tie=0.5."""
    from datasets import load_dataset

    ds = load_dataset("lmsys/mt_bench_human_judgments", split="human")
    if pair is None:
        pair = ("alpaca-13b", "gpt-3.5-turbo")
    a, b = pair
    rows = []
    for r in ds:
        if not ((r["model_a"] == a and r["model_b"] == b) or (r["model_a"] == b and r["model_b"] == a)):
            continue
        # Normalize so model_a is always `a`.
        flip = r["model_a"] != a
        ca, cb = (r["conversation_b"], r["conversation_a"]) if flip else (r["conversation_a"], r["conversation_b"])
        w = str(r["winner"])
        # winner is 'model_a' / 'model_b' / 'tie'; normalize so A is always `a`.
        if w == "tie":
            truth = 0.5
        elif (w == "model_a") != flip:
            truth = 1.0
        else:
            truth = 0.0
        qa = " ".join(m.get("content", "") for m in ca)
        qb = " ".join(m.get("content", "") for m in cb)
        rows.append({
            "item_id": f"mtbench-{r['question_id']}-{len(rows)}",
            "prompt": f"Question and two responses. A: {qa[:1500]}\nB: {qb[:1500]}",
            "human": truth,
            "truth_raw": str(w),
            "meta": {"kind": "pairwise"},
        })
        if len(rows) >= max_n:
            break
    return rows


def load_arena(n: int = 800, seed: int = 0) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("lmarena-ai/arena-human-preference-55k", split="train", streaming=True)
    rng = np.random.default_rng(seed)
    rows = []
    for r in ds:
        if rng.random() > n / 55000:
            continue
        wa, wb, wt = int(r["winner_model_a"]), int(r["winner_model_b"]), int(r["winner_tie"])
        truth = 0.5 if wt else float(wa)
        rows.append({
            "item_id": f"arena-{r['id']}",
            "prompt": f"Prompt: {str(r['prompt'])[:1200]}\nA: {str(r['response_a'])[:1500]}\nB: {str(r['response_b'])[:1500]}",
            "human": truth,
            "truth_raw": truth,
            "meta": {"kind": "pairwise"},
        })
        if len(rows) >= n:
            break
    return rows


def load_mqm(system: str | None = "Online-W", max_n: int = 500) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("RicardoRei/wmt-mqm-human-evaluation", split="train", streaming=True)
    # Pick the most frequent en-de system at build time if not given.
    rows = []
    for r in ds:
        if r["lp"] != "en-de":
            continue
        if system is not None and r["system"] != system:
            continue
        rows.append({
            "item_id": f"mqm-{r['year']}-{r['system']}-{len(rows)}",
            "prompt": f"Source: {r['src']}\nTranslation: {r['mt']}",
            "human": float(r["score"]),
            "truth_raw": float(r["score"]),
            "meta": {"kind": "real", "system": r["system"]},
            "system": r["system"],
        })
        if len(rows) >= max_n and system is not None:
            break
    if system is None:
        # Most frequent system, then filter.
        from collections import Counter

        top = Counter(r["system"] for r in rows).most_common(1)[0][0]
        rows = [r for r in rows if r["system"] == top][:max_n]
    return rows


def load_toxicity(n: int = 800, seed: int = 0) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("google/civil_comments", split="train", streaming=True)
    rng = np.random.default_rng(seed)
    rows = []
    # Oversample toxic (rare ~8%) for a balanced-ish slice: take all toxic + fill.
    buf: list[dict[str, Any]] = []
    tox: list[dict[str, Any]] = []
    for r in ds:
        t = float(r["toxicity"]) >= 0.5
        row = {
            "item_id": f"civil-{len(buf) + len(tox)}",
            "prompt": f"Comment: {r['text'][:1500]}",
            "human": float(t),
            "truth_raw": float(r["toxicity"]),
            "meta": {"kind": "binary", "labels": ["non-toxic", "toxic"]},
        }
        (tox if t else buf).append(row)
        if len(tox) >= n // 2 and len(buf) >= n:
            break
    want_tox = min(len(tox), n // 2)
    rows = tox[:want_tox] + buf[: n - want_tox]
    rng.shuffle(rows)
    return rows


def load_oraclebench(n: int = 800, seed: int = 0) -> list[dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("raihan-js/oraclebench-items", "items.jsonl", repo_type="dataset")
    import json

    all_rows = [json.loads(line) for line in open(p) if line.strip()]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_rows), min(n, len(all_rows)), replace=False)
    rows = []
    for k in sorted(idx):
        r = all_rows[int(k)]
        truth = 1.0 if str(r["oracle_verdict"]).lower() in ("correct", "true", "1", "yes") else 0.0
        rows.append({
            "item_id": f"oracle-{k}",
            "prompt": f"Question: {r['prompt'][:1500]}\nProposed answer: {r['answer'][:1500]}",
            "human": truth,
            "truth_raw": str(r["oracle_verdict"]),
            "meta": {"kind": "binary", "labels": ["incorrect", "correct"]},
        })
    return rows


FEDPROC_ROLES = [
    "INCORPORATED_BY_REFERENCE", "INTERNAL_REFERENCE", "CHECKLIST_NOT_SELECTED",
    "FULL_TEXT", "CHECKLIST_SELECTED", "NARRATIVE_MENTION", "INDEX_ENTRY",
    "NOT_A_CLAUSE", "EXPLICITLY_EXCLUDED",
]


def load_fedproc(n: int = 800, seed: int = 0) -> list[dict[str, Any]]:
    """800 audited mentions; judge predicts the role (categorical)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    gold = pd.read_parquet("/home/raihan/Desktop/APPS/research-learn/fedproc-ledger/data/interim/labeled_mentions.parquet")
    mp = hf_hub_download("raihan-js/fedproc-ledger-bench", "mentions.parquet", repo_type="dataset")
    mentions = pd.read_parquet(mp, columns=["doc_id", "number", "context"])
    gold = gold.sample(n=min(n, len(gold)), random_state=seed).reset_index(drop=True)
    rows = []
    for _, g in gold.iterrows():
        ctx = mentions[mentions["doc_id"] == g["doc"]]
        context = str(ctx.iloc[0]["context"])[:1200] if len(ctx) else ""
        rows.append({
            "item_id": f"fedproc-{g['id']}",
            "prompt": f"Clause mention context: {context}\nClassify the role of this mention.",
            "human": str(g["label"]),
            "truth_raw": str(g["label"]),
            "meta": {"kind": "categorical", "labels": FEDPROC_ROLES},
        })
    return rows


LOADERS = {
    "helpsteer2-correctness": load_helpsteer2,
    "summeval-consistency": lambda n=100, seed=0: load_summeval(),
    "mtbench-winrate": lambda n=600, seed=0: load_mtbench(),
    "arena-winrate": load_arena,
    "wmt-mqm": lambda n=500, seed=0: load_mqm(max_n=n),
    "toxicity-flag": load_toxicity,
    "oraclebench-truth": load_oraclebench,
    "fedproc-roles": load_fedproc,
}
