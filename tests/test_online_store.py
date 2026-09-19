"""Unit tests for the low-latency online store."""

from __future__ import annotations

import concurrent.futures
import time
from datetime import datetime, timedelta, timezone
import pytest
from featurehub.online.store import OnlineStore


def test_online_store_write_and_read():
    store = OnlineStore(num_shards=8)
    now = datetime.now(timezone.utc)

    store.write_features("user_view", "user_1", {"f1": 42, "f2": "premium"}, now)

    # Single entity retrieval
    res = store.get_online_features(["user_1"], ["user_view:f1", "user_view:f2"])
    assert len(res) == 1
    assert res[0]["entity_key"] == "user_1"
    assert res[0]["user_view:f1"] == 42
    assert res[0]["user_view:f2"] == "premium"


def test_online_store_batch_mget():
    store = OnlineStore(num_shards=16)
    records = []
    now = datetime.now(timezone.utc)

    for i in range(100):
        records.append((f"user_{i}", {"score": i * 1.5}, now))
    store.write_batch("user_view", records)

    assert store.total_records() == 100

    # Batch read 50 keys
    keys = [f"user_{i}" for i in range(50)]
    results = store.get_online_features(keys, ["user_view:score"])
    assert len(results) == 50
    for i, r in enumerate(results):
        assert r["entity_key"] == f"user_{i}"
        assert r["user_view:score"] == i * 1.5


def test_online_store_ttl_expiration():
    store = OnlineStore(num_shards=4)
    store.register_feature_view_ttl("user_view", ttl_seconds=2)

    past_time = datetime.now(timezone.utc) - timedelta(seconds=5)
    store.write_features("user_view", "user_old", {"score": 100}, past_time)

    # Expired feature should return None
    res = store.get_online_features(["user_old"], ["user_view:score"])
    assert res[0]["user_view:score"] is None

    # Fresh feature
    fresh_time = datetime.now(timezone.utc)
    store.write_features("user_view", "user_fresh", {"score": 200}, fresh_time)
    res_fresh = store.get_online_features(["user_fresh"], ["user_view:score"])
    assert res_fresh[0]["user_view:score"] == 200


def test_online_store_concurrency_and_latency():
    store = OnlineStore(num_shards=32)
    now = datetime.now(timezone.utc)

    # Pre-populate 500 keys
    for i in range(500):
        store.write_features("user_view", f"u_{i}", {"f": i}, now)

    # Concurrently read across 10 threads, 2000 total queries
    def worker(worker_id: int):
        for j in range(200):
            key = f"u_{(worker_id * 20 + j) % 500}"
            r = store.get_online_features([key], ["user_view:f"])
            assert len(r) == 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(worker, w) for w in range(10)]
        concurrent.futures.wait(futures)

    stats = store.read_telemetry.get_stats()
    assert stats["count"] == 2000
    # Assert p99 latency is well under 10ms in Python environment (typically < 0.5ms)
    assert stats["p99_ms"] < 10.0
