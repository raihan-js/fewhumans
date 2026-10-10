"""Runner tests — fully offline via httpx.MockTransport (no network)."""

import asyncio
import json

import httpx
import pytest

from fewhumans.judges.runner import (
    CostCapExceeded,
    JudgeCall,
    JudgeConfig,
    PriceTable,
    Runner,
    parse_output,
)


def _openai_transport(text="{\"label\": \"correct\", \"confidence\": 0.9}", tin=100, tout=20):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": tin, "completion_tokens": tout},
        })
    return httpx.MockTransport(handler)


def _runner(text="{\"label\": \"correct\", \"confidence\": 0.9}", tmp_path=None, **kw):
    prices = PriceTable(per_1k_in={"m": 0.15}, per_1k_out={"m": 0.60})
    client = httpx.AsyncClient(transport=_openai_transport(text), base_url="http://t")
    return Runner(prices, cache_dir=str(tmp_path or "/tmp/fhcache"), client=client), prices


def test_parse_ok_and_abstain():
    lab, conf, st = parse_output('... {"label": "a", "confidence": 0.7} ...')
    assert (lab, conf, st) == ("a", 0.7, "ok")
    lab, conf, st = parse_output("no json here")
    assert (lab, conf, st) == ("ABSTAIN", None, "ABSTAIN")


def test_judge_one_caches_and_costs(tmp_path):
    runner, _ = _runner(tmp_path=tmp_path)
    cfg = JudgeConfig(name="j", model="m", base_url="http://t")
    c1 = asyncio.run(runner.judge_one(cfg, "i1", "prompt", 5.0))
    assert isinstance(c1, JudgeCall) and c1.label == "correct" and c1.confidence == 0.9
    assert abs(c1.cost_usd - (100 / 1000 * 0.15 + 20 / 1000 * 0.60)) < 1e-12
    # Second call is a cache hit even if the transport would now fail.
    runner.client = httpx.AsyncClient(transport=_openai_transport("GARBAGE"), base_url="http://t")
    c2 = asyncio.run(runner.judge_one(cfg, "i1", "prompt", 5.0))
    assert c2.label == "correct" and runner.spent_usd == c1.cost_usd


def test_abstain_recorded_not_coerced(tmp_path):
    runner, _ = _runner("total garbage", tmp_path=tmp_path)
    cfg = JudgeConfig(name="j", model="m", base_url="http://t")
    c = asyncio.run(runner.judge_one(cfg, "i9", "prompt", 5.0))
    assert c.label == "ABSTAIN" and c.parse_status == "ABSTAIN" and c.raw == "total garbage"


def test_cost_cap_stops(tmp_path):
    runner, _ = _runner(tmp_path=tmp_path)
    cfg = JudgeConfig(name="j", model="m", base_url="http://t")
    items = [(f"i{k}", "prompt") for k in range(5)]
    with pytest.raises(CostCapExceeded):
        asyncio.run(runner.run_all(cfg, items, max_cost_usd=0.05))
    # Partial results were cached.
    assert list(tmp_path.rglob("*.json")), "expected cached partial calls"


def test_self_consistency_majority(tmp_path):
    texts = [
        "{\"label\": \"a\", \"confidence\": 0.9}",
        "{\"label\": \"b\", \"confidence\": 0.8}",
        "{\"label\": \"a\", \"confidence\": 0.7}",
    ]
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        t = texts[calls["n"] % 3]
        calls["n"] += 1
        return httpx.Response(200, json={
            "choices": [{"message": {"content": t}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })

    prices = PriceTable(per_1k_in={"m": 0.0}, per_1k_out={"m": 0.0})
    runner = Runner(prices, cache_dir=str(tmp_path), client=httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://t"))
    cfg = JudgeConfig(name="j", model="m", base_url="http://t", confidence="self_consistency", self_consistency_k=3)
    c = asyncio.run(runner.judge_one(cfg, "i1", "prompt", 5.0))
    assert c.label == "a" and abs(c.confidence - 2 / 3) < 1e-12 and len(c.samples) == 3
    assert json.loads(c.raw)["label"] == "a"
