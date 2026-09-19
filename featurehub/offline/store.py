"""DuckDB-powered offline feature store.

Provides point-in-time correct historical feature joins using native ASOF joins,
guaranteeing zero temporal leakage and eliminating lookahead bias for ML model training.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import duckdb
import pandas as pd


class OfflineStore:
    """Columnar offline store backed by DuckDB for time-travel and point-in-time joins."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        # Use a thread-safe connection pattern
        self._lock = threading.Lock()
        self.conn = duckdb.connect(database=db_path)
        self._registered_views: set[str] = set()

    def _ensure_view_table(self, feature_view: str, sample_row: Dict[str, Any]):
        """Ensures the DuckDB table exists for the given feature view."""
        with self._lock:
            if feature_view in self._registered_views:
                return

            # Check if table already exists in DuckDB
            table_check = self.conn.execute(
                f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{feature_view}'"
            ).fetchone()[0]

            if table_check == 0:
                # Infer columns from sample row
                col_defs = ["entity_key VARCHAR", "event_timestamp TIMESTAMP WITH TIME ZONE", "created_timestamp TIMESTAMP WITH TIME ZONE"]
                for k, v in sample_row.items():
                    if k in ("entity_key", "event_timestamp", "created_timestamp"):
                        continue
                    if isinstance(v, int):
                        col_type = "BIGINT"
                    elif isinstance(v, float):
                        col_type = "DOUBLE"
                    elif isinstance(v, bool):
                        col_type = "BOOLEAN"
                    else:
                        col_type = "VARCHAR"
                    col_defs.append(f'"{k}" {col_type}')

                create_sql = f"CREATE TABLE IF NOT EXISTS {feature_view} ({', '.join(col_defs)});"
                self.conn.execute(create_sql)

            self._registered_views.add(feature_view)

    def write_features_df(self, feature_view: str, df: pd.DataFrame):
        """Writes a batch DataFrame of historical features into the offline store.

        Expects columns: 'entity_key', 'event_timestamp', and feature columns.
        """
        if df.empty:
            return

        df_copy = df.copy()
        if "created_timestamp" not in df_copy.columns:
            df_copy["created_timestamp"] = datetime.now(timezone.utc)

        # Convert timestamps to datetime if strings
        df_copy["event_timestamp"] = pd.to_datetime(df_copy["event_timestamp"])
        df_copy["created_timestamp"] = pd.to_datetime(df_copy["created_timestamp"])
        df_copy["entity_key"] = df_copy["entity_key"].astype(str)

        sample = df_copy.iloc[0].to_dict()
        self._ensure_view_table(feature_view, sample)

        with self._lock:
            # Register temp view and insert
            self.conn.register("df_to_insert", df_copy)
            # Find column intersection (PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk)
            existing_cols = {
                row[1] for row in self.conn.execute(f"PRAGMA table_info('{feature_view}')").fetchall()
            }
            # Dynamically add any newly introduced columns to DuckDB table
            for c in df_copy.columns:
                if c not in existing_cols:
                    sample_val = df_copy[c].dropna().iloc[0] if not df_copy[c].dropna().empty else None
                    if isinstance(sample_val, (int,)):
                        col_type = "BIGINT"
                    elif isinstance(sample_val, (float,)):
                        col_type = "DOUBLE"
                    elif isinstance(sample_val, bool):
                        col_type = "BOOLEAN"
                    else:
                        col_type = "VARCHAR"
                    self.conn.execute(f'ALTER TABLE {feature_view} ADD COLUMN "{c}" {col_type}')
                    existing_cols.add(c)

            common_cols = [c for c in df_copy.columns if c in existing_cols]
            cols_str = ", ".join([f'"{c}"' for c in common_cols])

            insert_sql = f"""
                INSERT INTO {feature_view} ({cols_str})
                SELECT {cols_str} FROM df_to_insert
            """
            self.conn.execute(insert_sql)
            self.conn.unregister("df_to_insert")

    def write_record(
        self,
        feature_view: str,
        entity_key: str,
        features: Dict[str, Any],
        event_timestamp: Optional[datetime] = None,
        created_timestamp: Optional[datetime] = None,
    ):
        """Appends a single historical event record."""
        ts_event = event_timestamp or datetime.now(timezone.utc)
        ts_created = created_timestamp or datetime.now(timezone.utc)
        record = {
            "entity_key": str(entity_key),
            "event_timestamp": ts_event,
            "created_timestamp": ts_created,
            **features,
        }
        df = pd.DataFrame([record])
        self.write_features_df(feature_view, df)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: List[str],
        ttl_seconds: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        """Executes point-in-time (ASOF) joins between entity observations and historical features.

        entity_df must contain:
            - 'entity_key': string or convertible
            - 'event_timestamp': timestamp of the observation / target event

        features: list of feature references in "feature_view:feature_name" format.
        ttl_seconds: optional dict mapping feature_view_name -> max lookback seconds.

        Returns:
            pd.DataFrame containing entity_df plus all requested features as of event_timestamp.
        """
        if entity_df.empty:
            return entity_df.copy()

        obs_df = entity_df.copy()
        obs_df["entity_key"] = obs_df["entity_key"].astype(str)
        obs_df["event_timestamp"] = pd.to_datetime(obs_df["event_timestamp"])
        obs_df["_obs_row_id"] = range(len(obs_df))

        # Parse requested features by feature view
        fv_to_features: Dict[str, List[str]] = {}
        for feat in features:
            if ":" in feat:
                fv, f_name = feat.split(":", 1)
            else:
                fv, f_name = "default", feat
            fv_to_features.setdefault(fv, []).append(f_name)

        ttl_map = ttl_seconds or {}

        with self._lock:
            self.conn.register("obs_table", obs_df)
            current_table = "obs_table"

            for fv, f_names in fv_to_features.items():
                if fv not in self._registered_views:
                    # Table might exist or view has no data yet
                    table_exists = self.conn.execute(
                        f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{fv}'"
                    ).fetchone()[0] > 0
                    if not table_exists:
                        # Add null columns for this view
                        for f in f_names:
                            obs_df[f"{fv}:{f}"] = None
                        continue
                    self._registered_views.add(fv)

                # DuckDB ASOF JOIN:
                # obs.event_timestamp >= fv.event_timestamp finds the latest feature row <= observation time
                target_selects = [f'{current_table}.*']
                for f in f_names:
                    target_selects.append(f'"{fv}"."{f}" AS "{fv}:{f}"')
                if ttl_map.get(fv) is not None:
                    # Check TTL condition: obs.event_timestamp - fv.event_timestamp <= ttl
                    # We can use CASE WHEN to nullify expired features
                    ttl_sec = ttl_map[fv]
                    target_selects = [f'{current_table}.*']
                    for f in f_names:
                        target_selects.append(
                            f"""CASE 
                                WHEN epoch("{current_table}"."event_timestamp") - epoch("{fv}"."event_timestamp") <= {ttl_sec} 
                                THEN "{fv}"."{f}" 
                                ELSE NULL 
                            END AS "{fv}:{f}" """
                        )

                select_clause = ", ".join(target_selects)

                asof_sql = f"""
                    CREATE TEMP TABLE _joined_{fv} AS
                    SELECT {select_clause}
                    FROM {current_table}
                    ASOF LEFT JOIN {fv}
                        ON {current_table}.entity_key = {fv}.entity_key
                        AND {current_table}.event_timestamp >= {fv}.event_timestamp
                """
                self.conn.execute(asof_sql)

                if current_table != "obs_table":
                    self.conn.execute(f"DROP TABLE {current_table}")
                current_table = f"_joined_{fv}"

            result_df = self.conn.execute(f"SELECT * FROM {current_table} ORDER BY _obs_row_id").df()

            if current_table != "obs_table":
                self.conn.execute(f"DROP TABLE {current_table}")
            self.conn.unregister("obs_table")

        result_df.drop(columns=["_obs_row_id"], inplace=True, errors="ignore")
        return result_df

    def get_timeline(self, entity_key: str, feature_view: str) -> List[Dict[str, Any]]:
        """Retrieves complete historical timeline of feature updates for an entity."""
        with self._lock:
            table_exists = self.conn.execute(
                f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{feature_view}'"
            ).fetchone()[0] > 0
            if not table_exists:
                return []

            sql = f"""
                SELECT * FROM {feature_view}
                WHERE entity_key = '{entity_key}'
                ORDER BY event_timestamp ASC
            """
            df = self.conn.execute(sql).df()
            return df.to_dict(orient="records")

    def count_records(self, feature_view: str) -> int:
        with self._lock:
            table_exists = self.conn.execute(
                f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{feature_view}'"
            ).fetchone()[0] > 0
            if not table_exists:
                return 0
            return self.conn.execute(f"SELECT count(*) FROM {feature_view}").fetchone()[0]
