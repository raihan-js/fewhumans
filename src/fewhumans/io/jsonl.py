"""JSONL/CSV import-export + Label Studio thin adapter (spec §5.3)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from fewhumans.core.items import Item


def read_items_jsonl(path: str | Path) -> list[Item]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(Item.from_dict(json.loads(line)))
    return items


def write_items_jsonl(items: list[Item], path: str | Path) -> None:
    with open(path, "w") as f:
        f.writelines(json.dumps(it.to_dict()) + "\n" for it in items)


def read_labels_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_labels_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def read_items_csv(path: str | Path) -> list[Item]:
    items = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            items.append(Item(item_id=row["item_id"], payload=row.get("payload", row.get("text", ""))))
    return items
