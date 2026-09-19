"""Unified FeatureStore client for FeatureHub."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union
import pandas as pd

from featurehub.metadata import DataType, Entity, Feature, FeatureRecord, FeatureView
from featurehub.offline.store import OfflineStore
from featurehub.online.store import OnlineStore
from featurehub.registry import Registry
from featurehub.skew import SkewParityReport, SkewValidator
from featurehub.transformations import FeatureTransformation, TransformationRegistry


class FeatureStore:
    """The central unified interface for FeatureHub.

    Provides online low-latency serving, point-in-time offline historical joins,
    and consistency guarantees between training and inference.
    """

    def __init__(self, offline_db_path: str = ":memory:", num_online_shards: int = 32):
        self.registry = Registry()
        self.transformations = TransformationRegistry()
        self.online_store = OnlineStore(num_shards=num_online_shards)
        self.offline_store = OfflineStore(db_path=offline_db_path)
        self.skew_validator = SkewValidator()

    # --- Registration ---

    def register_entity(self, entity: Entity) -> Entity:
        return self.registry.register_entity(entity)

    def register_feature_view(self, feature_view: FeatureView) -> FeatureView:
        fv = self.registry.register_feature_view(feature_view)
        if fv.online and fv.ttl_seconds is not None:
            self.online_store.register_feature_view_ttl(fv.name, fv.ttl_seconds)
        return fv

    def register_transformation(self, transformation: FeatureTransformation):
        self.transformations.register(transformation)

    # --- Ingestion ---

    def ingest_stream(
        self,
        feature_view_name: str,
        entity_key: str,
        features: Dict[str, Any],
        event_timestamp: Optional[datetime] = None,
    ):
        """Streaming ingestion: writes to online store with sub-millisecond latency

        and appends to offline store for immutable historical time-travel.
        """
        fv = self.registry.get_feature_view(feature_view_name)
        if not fv:
            raise ValueError(f"FeatureView '{feature_view_name}' is not registered.")

        ts = event_timestamp or datetime.now(timezone.utc)

        # Apply unified transformation if configured
        final_features = dict(features)
        if fv.transformation_name:
            trans = self.transformations.get(fv.transformation_name)
            if trans:
                final_features = trans.transform_record(final_features)

        # Write to Online Store for real-time serving
        if fv.online:
            self.online_store.write_features(feature_view_name, str(entity_key), final_features, ts)

        # Write to Offline Store for historical point-in-time joins
        if fv.offline:
            self.offline_store.write_record(feature_view_name, str(entity_key), final_features, ts)

    def ingest_batch(
        self,
        feature_view_name: str,
        df: pd.DataFrame,
        sync_to_online: bool = True,
    ):
        """Batch ingestion: writes historical DataFrame to offline store

        and syncs the latest feature state per entity to the online store.
        """
        fv = self.registry.get_feature_view(feature_view_name)
        if not fv:
            raise ValueError(f"FeatureView '{feature_view_name}' is not registered.")

        df_processed = df.copy()

        # Apply unified transformation if configured
        if fv.transformation_name:
            trans = self.transformations.get(fv.transformation_name)
            if trans:
                df_processed = trans.transform_batch(df_processed)

        # Write to Offline Store
        if fv.offline:
            self.offline_store.write_features_df(feature_view_name, df_processed)

        # Sync latest state to Online Store
        if fv.online and sync_to_online and not df_processed.empty:
            df_sorted = df_processed.sort_values(by="event_timestamp", ascending=True)
            latest_per_entity = df_sorted.groupby("entity_key").last().reset_index()

            batch_records = []
            feature_cols = [
                c for c in latest_per_entity.columns
                if c not in ("entity_key", "event_timestamp", "created_timestamp")
            ]
            for _, row in latest_per_entity.iterrows():
                f_dict = {col: row[col] for col in feature_cols}
                ts = row["event_timestamp"]
                batch_records.append((str(row["entity_key"]), f_dict, ts))

            self.online_store.write_batch(feature_view_name, batch_records)

    # --- Serving ---

    def get_online_features(
        self,
        entity_keys: List[str],
        features: List[str],
        log_for_skew_check: bool = True,
    ) -> List[Dict[str, Any]]:
        """Ultra-low latency batch retrieval for real-time inference.

        Optionally logs inference queries to verify 0% training-serving skew.
        """
        results = self.online_store.get_online_features(entity_keys, features)

        if log_for_skew_check:
            now = datetime.now(timezone.utc)
            for row in results:
                self.skew_validator.log_inference_event(
                    entity_key=row["entity_key"],
                    features=row,
                    timestamp=now,
                )

        return results

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: List[str],
    ) -> pd.DataFrame:
        """Executes point-in-time (ASOF) joins to construct training datasets with zero lookahead leakage."""
        ttl_map = {}
        for feat in features:
            fv_name = feat.split(":", 1)[0] if ":" in feat else "default"
            fv = self.registry.get_feature_view(fv_name)
            if fv and fv.ttl_seconds is not None:
                ttl_map[fv_name] = fv.ttl_seconds

        return self.offline_store.get_historical_features(entity_df, features, ttl_seconds=ttl_map)

    # --- Skew Verification ---

    def verify_skew(
        self,
        features: List[str],
        sample_size: int = 500,
        float_tolerance: float = 1e-5,
    ) -> SkewParityReport:
        """Verifies parity between logged online inferences and offline historical point-in-time queries."""
        return self.skew_validator.verify_parity(
            offline_store=self.offline_store,
            feature_refs=features,
            sample_size=sample_size,
            float_tolerance=float_tolerance,
        )

    # --- Telemetry & Status ---

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "online": self.online_store.get_telemetry(),
            "registry": self.registry.to_dict(),
        }
