"""Unit tests for training-serving skew elimination and parity verification."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pandas as pd
import pytest

from featurehub.client import FeatureStore
from featurehub.metadata import DataType, Entity, Feature, FeatureView
from featurehub.transformations import FeatureTransformation


def test_zero_training_serving_skew_parity():
    store = FeatureStore(offline_db_path=":memory:", num_online_shards=8)

    # 1. Register Schema
    entity = Entity(name="user", join_keys=["user_id"])
    store.register_entity(entity)

    # Unified Transformation
    def transform_fn(row):
        r = dict(row)
        r["ratio"] = round(float(r.get("val_a", 1)) / max(float(r.get("val_b", 1)), 0.001), 4)
        return r

    trans = FeatureTransformation(
        name="ratio_calc",
        input_fields=["val_a", "val_b"],
        output_fields=["ratio"],
        func=transform_fn,
    )
    store.register_transformation(trans)

    fv = FeatureView(
        name="skew_test_view",
        entity_name="user",
        features=[
            Feature(name="val_a", dtype=DataType.FLOAT64),
            Feature(name="val_b", dtype=DataType.FLOAT64),
            Feature(name="ratio", dtype=DataType.FLOAT64),
        ],
        online=True,
        offline=True,
        transformation_name="ratio_calc",
    )
    store.register_feature_view(fv)

    # 2. Ingest streaming records
    now = datetime.now(timezone.utc)
    for i in range(50):
        key = f"user_{i}"
        store.ingest_stream(
            feature_view_name="skew_test_view",
            entity_key=key,
            features={"val_a": 10.0 + i, "val_b": 2.0 + (i % 5)},
            event_timestamp=now - timedelta(seconds=100 - i),
        )

    # 3. Serve online features and log inference events
    feature_refs = ["skew_test_view:val_a", "skew_test_view:val_b", "skew_test_view:ratio"]
    for i in range(50):
        key = f"user_{i}"
        _ = store.get_online_features([key], feature_refs, log_for_skew_check=True)

    # 4. Verify Skew Parity
    report = store.verify_skew(feature_refs, sample_size=50)

    assert report.total_checks == 50 * 3  # 150 feature values checked
    assert report.mismatched_features == 0
    assert report.matched_features == 150
    assert report.skew_rate_percent == 0.0
    assert report.parity_rate_percent == 100.0


def test_skew_detector_catches_discrepancy():
    store = FeatureStore(offline_db_path=":memory:", num_online_shards=4)
    entity = Entity(name="user", join_keys=["user_id"])
    store.register_entity(entity)

    fv = FeatureView(
        name="test_fv",
        entity_name="user",
        features=[Feature(name="score", dtype=DataType.FLOAT64)],
        online=True,
        offline=True,
    )
    store.register_feature_view(fv)

    # Ingest event
    now = datetime.now(timezone.utc)
    store.ingest_stream("test_fv", "u_1", {"score": 10.0}, now)

    # Serve online
    _ = store.get_online_features(["u_1"], ["test_fv:score"], log_for_skew_check=True)

    # Deliberately modify offline store directly to introduce skew
    store.offline_store.conn.execute("UPDATE test_fv SET score = 99.0 WHERE entity_key = 'u_1'")

    # Verify skew validator catches the discrepancy
    report = store.verify_skew(["test_fv:score"], sample_size=1)
    assert report.mismatched_features == 1
    assert report.skew_rate_percent == 100.0
    assert len(report.mismatch_details) == 1
    assert report.mismatch_details[0]["online_value"] == 10.0
    assert report.mismatch_details[0]["offline_value"] == 99.0
