"""Unit tests for the DuckDB offline store and point-in-time ASOF joins."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pandas as pd
import pytest
from featurehub.offline.store import OfflineStore


def test_offline_store_asof_point_in_time():
    store = OfflineStore(db_path=":memory:")
    base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Ingest historical events for user_A at t = 10m, 20m, 30m
    events = [
        {"entity_key": "user_A", "event_timestamp": base_time + timedelta(minutes=10), "risk_score": 0.10},
        {"entity_key": "user_A", "event_timestamp": base_time + timedelta(minutes=20), "risk_score": 0.20},
        {"entity_key": "user_A", "event_timestamp": base_time + timedelta(minutes=30), "risk_score": 0.30},
    ]
    store.write_features_df("fraud_view", pd.DataFrame(events))

    # Test Observations:
    # Obs 1: before any event (t = 5m) -> should be NULL / None
    # Obs 2: between 10m and 20m (t = 15m) -> must be 0.10 (NEVER 0.20 or 0.30)
    # Obs 3: between 20m and 30m (t = 25m) -> must be 0.20 (NEVER 0.30)
    # Obs 4: after 30m (t = 35m) -> must be 0.30
    obs_df = pd.DataFrame([
        {"entity_key": "user_A", "event_timestamp": base_time + timedelta(minutes=5), "obs_id": 1},
        {"entity_key": "user_A", "event_timestamp": base_time + timedelta(minutes=15), "obs_id": 2},
        {"entity_key": "user_A", "event_timestamp": base_time + timedelta(minutes=25), "obs_id": 3},
        {"entity_key": "user_A", "event_timestamp": base_time + timedelta(minutes=35), "obs_id": 4},
    ])

    joined = store.get_historical_features(obs_df, ["fraud_view:risk_score"])

    res_1 = joined[joined["obs_id"] == 1]["fraud_view:risk_score"].iloc[0]
    res_2 = joined[joined["obs_id"] == 2]["fraud_view:risk_score"].iloc[0]
    res_3 = joined[joined["obs_id"] == 3]["fraud_view:risk_score"].iloc[0]
    res_4 = joined[joined["obs_id"] == 4]["fraud_view:risk_score"].iloc[0]

    assert pd.isna(res_1) or res_1 is None
    assert res_2 == 0.10  # Zero leakage! Future 0.20/0.30 is NOT leaked.
    assert res_3 == 0.20  # Zero leakage! Future 0.30 is NOT leaked.
    assert res_4 == 0.30


def test_offline_store_ttl_enforcement():
    store = OfflineStore(db_path=":memory:")
    base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Event at t = 0m
    events = [
        {"entity_key": "user_B", "event_timestamp": base_time, "feature_val": 99.0},
    ]
    store.write_features_df("ttl_view", pd.DataFrame(events))

    # Observation at t = 10m (lookback = 600s). TTL is 300s -> should expire to NULL
    obs_expired = pd.DataFrame([
        {"entity_key": "user_B", "event_timestamp": base_time + timedelta(seconds=600)},
    ])
    joined_expired = store.get_historical_features(
        obs_expired, ["ttl_view:feature_val"], ttl_seconds={"ttl_view": 300}
    )
    val_expired = joined_expired["ttl_view:feature_val"].iloc[0]
    assert pd.isna(val_expired) or val_expired is None

    # Observation at t = 3m (lookback = 180s). TTL is 300s -> should return 99.0
    obs_valid = pd.DataFrame([
        {"entity_key": "user_B", "event_timestamp": base_time + timedelta(seconds=180)},
    ])
    joined_valid = store.get_historical_features(
        obs_valid, ["ttl_view:feature_val"], ttl_seconds={"ttl_view": 300}
    )
    assert joined_valid["ttl_view:feature_val"].iloc[0] == 99.0
