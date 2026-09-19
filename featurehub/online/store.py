"""High-performance, low-latency online feature store.

Designed for sub-millisecond p99 latency key-value retrieval for real-time inference,
featuring sharded concurrency, batch mget, TTL expiration, and microsecond telemetry.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple


class LatencyTelemetry:
    """Thread-safe microsecond latency tracker with percentile computation."""

    def __init__(self, max_samples: int = 50000):
        self._samples: deque[float] = deque(maxlen=max_samples)
        self._lock = threading.Lock()
        self._total_requests = 0
        self._start_time = time.perf_counter()

    def record(self, duration_us: float):
        with self._lock:
            self._samples.append(duration_us)
            self._total_requests += 1

    def get_stats(self) -> Dict[str, float]:
        with self._lock:
            if not self._samples:
                return {
                    "count": 0,
                    "p50_ms": 0.0,
                    "p90_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                    "p999_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                    "avg_ms": 0.0,
                    "qps": 0.0,
                }

            sorted_samples = sorted(self._samples)
            n = len(sorted_samples)

            def percentile(p: float) -> float:
                idx = min(int(math.ceil(p * n)) - 1, n - 1)
                return sorted_samples[max(0, idx)]

            elapsed = max(time.perf_counter() - self._start_time, 0.001)
            qps = round(self._total_requests / elapsed, 2)

            return {
                "count": self._total_requests,
                "sample_size": n,
                "p50_ms": round(percentile(0.50) / 1000.0, 4),
                "p90_ms": round(percentile(0.90) / 1000.0, 4),
                "p95_ms": round(percentile(0.95) / 1000.0, 4),
                "p99_ms": round(percentile(0.99) / 1000.0, 4),
                "p999_ms": round(percentile(0.999) / 1000.0, 4),
                "min_ms": round(sorted_samples[0] / 1000.0, 4),
                "max_ms": round(sorted_samples[-1] / 1000.0, 4),
                "avg_ms": round((sum(sorted_samples) / n) / 1000.0, 4),
                "qps": qps,
            }

    def reset(self):
        with self._lock:
            self._samples.clear()
            self._total_requests = 0
            self._start_time = time.perf_counter()


class StoreShard:
    """A sharded memory segment to reduce lock contention under concurrent load."""

    def __init__(self):
        # key: (feature_view, entity_key) -> {feature_name: (value, timestamp_epoch)}
        self._data: Dict[Tuple[str, str], Dict[str, Tuple[Any, float]]] = {}
        self._lock = threading.RLock()

    def set_features(self, feature_view: str, entity_key: str, features: Dict[str, Any], timestamp_epoch: float):
        compound_key = (feature_view, entity_key)
        with self._lock:
            if compound_key not in self._data:
                self._data[compound_key] = {}
            target = self._data[compound_key]
            for f_name, f_val in features.items():
                target[f_name] = (f_val, timestamp_epoch)

    def get_features(
        self,
        feature_view: str,
        entity_key: str,
        feature_names: Optional[List[str]],
        ttl_seconds: Optional[int],
        now_epoch: float,
    ) -> Dict[str, Any]:
        compound_key = (feature_view, entity_key)
        with self._lock:
            record = self._data.get(compound_key)
            if not record:
                return {}

            result = {}
            names_to_fetch = feature_names if feature_names is not None else record.keys()
            for name in names_to_fetch:
                item = record.get(name)
                if item is not None:
                    val, ts = item
                    if ttl_seconds is not None and (now_epoch - ts) > ttl_seconds:
                        continue  # Expired
                    result[name] = val
            return result

    def get_all_for_entity(self, entity_key: str) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            result = {}
            for (fv, ek), features in self._data.items():
                if ek == entity_key:
                    result[fv] = {k: v[0] for k, v in features.items()}
            return result

    def count_keys(self) -> int:
        with self._lock:
            return len(self._data)


class OnlineStore:
    """Production-grade, low-latency online feature store."""

    def __init__(self, num_shards: int = 32):
        self.num_shards = num_shards
        self.shards = [StoreShard() for _ in range(num_shards)]
        self.read_telemetry = LatencyTelemetry()
        self.write_telemetry = LatencyTelemetry()
        self._ttl_registry: Dict[str, Optional[int]] = {}

    def _get_shard(self, entity_key: str) -> StoreShard:
        shard_idx = hash(entity_key) % self.num_shards
        return self.shards[shard_idx]

    def register_feature_view_ttl(self, feature_view: str, ttl_seconds: Optional[int]):
        self._ttl_registry[feature_view] = ttl_seconds

    def write_features(
        self,
        feature_view: str,
        entity_key: str,
        features: Dict[str, Any],
        event_timestamp: Optional[datetime] = None,
    ):
        """Writes/updates online features for an entity with microsecond tracking."""
        t0 = time.perf_counter_ns()
        ts_epoch = event_timestamp.timestamp() if event_timestamp else time.time()
        shard = self._get_shard(entity_key)
        shard.set_features(feature_view, entity_key, features, ts_epoch)
        duration_us = (time.perf_counter_ns() - t0) / 1000.0
        self.write_telemetry.record(duration_us)

    def write_batch(
        self,
        feature_view: str,
        records: List[Tuple[str, Dict[str, Any], Optional[datetime]]],
    ):
        """Batch write features."""
        t0 = time.perf_counter_ns()
        for entity_key, features, event_timestamp in records:
            ts_epoch = event_timestamp.timestamp() if event_timestamp else time.time()
            shard = self._get_shard(entity_key)
            shard.set_features(feature_view, entity_key, features, ts_epoch)
        duration_us = (time.perf_counter_ns() - t0) / 1000.0
        self.write_telemetry.record(duration_us)

    def get_online_features(
        self,
        entity_keys: List[str],
        features: List[str],
    ) -> List[Dict[str, Any]]:
        """Ultra-low latency batch feature lookup.

        Args:
            entity_keys: List of entity IDs, e.g. ["user_1001", "user_1002"]
            features: List of feature references in "feature_view:feature_name" format,
                      e.g. ["user_view:fraud_score", "user_view:tx_count_10m"]

        Returns:
            List of dicts: [{"entity_key": "user_1001", "user_view:fraud_score": 0.12, ...}, ...]
        """
        t0 = time.perf_counter_ns()
        now_epoch = time.time()

        # Parse feature references
        # Group requested features by feature_view
        fv_to_features: Dict[str, List[str]] = {}
        for feat in features:
            if ":" in feat:
                fv, f_name = feat.split(":", 1)
            else:
                fv, f_name = "default", feat
            fv_to_features.setdefault(fv, []).append(f_name)

        results: List[Dict[str, Any]] = []

        for entity_key in entity_keys:
            shard = self._get_shard(entity_key)
            row: Dict[str, Any] = {"entity_key": entity_key}

            for fv, f_names in fv_to_features.items():
                ttl = self._ttl_registry.get(fv)
                extracted = shard.get_features(fv, entity_key, f_names, ttl, now_epoch)
                for f_name in f_names:
                    full_ref = f"{fv}:{f_name}"
                    row[full_ref] = extracted.get(f_name, None)

            results.append(row)

        duration_us = (time.perf_counter_ns() - t0) / 1000.0
        self.read_telemetry.record(duration_us)
        return results

    def get_entity_all(self, entity_key: str) -> Dict[str, Dict[str, Any]]:
        shard = self._get_shard(entity_key)
        return shard.get_all_for_entity(entity_key)

    def total_records(self) -> int:
        return sum(shard.count_keys() for shard in self.shards)

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "read": self.read_telemetry.get_stats(),
            "write": self.write_telemetry.get_stats(),
            "total_keys": self.total_records(),
        }
