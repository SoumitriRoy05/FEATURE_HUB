"""Real-World Production Scenario: Multi-Domain Enterprise Feature Store.

Simulates:
1. User fraud features (velocity, risk score, trust)
2. Merchant risk features (chargeback rate, category risk, volume)
3. Device fingerprint features (browser anomaly, headless flag, device age)
4. Customer lifetime value features (total orders, churn probability, avg basket)
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Ensure workspace root is in sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from featurehub.client import FeatureStore
from featurehub.metadata import DataType, Entity, Feature, FeatureView
from featurehub.transformations import FeatureTransformation


def setup_fraud_feature_store(store: FeatureStore) -> FeatureStore:
    """Configures the FeatureHub store with entities, feature views, and unified transformations."""

    # 1. Register Entities
    user_entity = Entity(
        name="user",
        join_keys=["user_id"],
        description="Banking and e-commerce end user account",
    )
    store.register_entity(user_entity)

    merchant_entity = Entity(
        name="merchant",
        join_keys=["merchant_id"],
        description="Payment gateway merchant processing transactions",
    )
    store.register_entity(merchant_entity)

    device_entity = Entity(
        name="device",
        join_keys=["device_id"],
        description="Hardware and browser fingerprint profile",
    )
    store.register_entity(device_entity)

    # 2. Register Unified Transformations
    def fraud_transformation_func(record: Dict[str, any]) -> Dict[str, any]:
        res = dict(record)
        c10 = res.get("tx_count_10m", 0) or 0
        c1h = res.get("tx_count_1h", 1) or 1
        res["velocity_ratio"] = round(float(c10) / max(float(c1h), 1.0), 4)

        risk = res.get("risk_score", 0.1) or 0.1
        trust = res.get("device_trust_score", 0.9) or 0.9
        res["composite_risk"] = round(float(risk) * (1.0 - float(trust) * 0.5), 4)
        return res

    def fraud_batch_func(df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        c10 = df_out["tx_count_10m"].fillna(0).astype(float)
        c1h = df_out["tx_count_1h"].fillna(1).astype(float).clip(lower=1.0)
        df_out["velocity_ratio"] = (c10 / c1h).round(4)

        risk = df_out["risk_score"].fillna(0.1).astype(float)
        trust = df_out["device_trust_score"].fillna(0.9).astype(float)
        df_out["composite_risk"] = (risk * (1.0 - trust * 0.5)).round(4)
        return df_out

    trans_fraud = FeatureTransformation(
        name="fraud_ratios",
        input_fields=["tx_count_10m", "tx_count_1h", "risk_score", "device_trust_score"],
        output_fields=["velocity_ratio", "composite_risk"],
        func=fraud_transformation_func,
        batch_func=fraud_batch_func,
        description="Calculates transaction velocity ratios and trust-adjusted composite risk",
    )
    store.register_transformation(trans_fraud)

    # Merchant Transformation
    def merchant_trans_func(record: Dict[str, any]) -> Dict[str, any]:
        res = dict(record)
        cb_rate = res.get("chargeback_rate", 0.01) or 0.01
        vol = res.get("volume_30d", 10000.0) or 10000.0
        # High exposure flag
        res["exposure_score"] = round(float(cb_rate) * (float(vol) ** 0.5) / 100.0, 4)
        return res

    def merchant_batch_func(df: pd.DataFrame) -> pd.DataFrame:
        df_out = df.copy()
        cb = df_out["chargeback_rate"].fillna(0.01).astype(float)
        vol = df_out["volume_30d"].fillna(10000.0).astype(float)
        df_out["exposure_score"] = (cb * (vol ** 0.5) / 100.0).round(4)
        return df_out

    trans_merchant = FeatureTransformation(
        name="merchant_exposure",
        input_fields=["chargeback_rate", "volume_30d"],
        output_fields=["exposure_score"],
        func=merchant_trans_func,
        batch_func=merchant_batch_func,
        description="Calculates volume-weighted merchant exposure score",
    )
    store.register_transformation(trans_merchant)

    # 3. Register Feature Views

    # View 1: User Fraud Features
    user_features = FeatureView(
        name="user_fraud_features",
        entity_name="user",
        features=[
            Feature(name="tx_count_10m", dtype=DataType.INT64, description="Transactions in last 10 minutes"),
            Feature(name="tx_count_1h", dtype=DataType.INT64, description="Transactions in last 1 hour"),
            Feature(name="tx_amount_1h", dtype=DataType.FLOAT64, description="Total transaction amount in last 1 hour"),
            Feature(name="risk_score", dtype=DataType.FLOAT64, description="Machine learning base risk score (0.0 to 1.0)"),
            Feature(name="device_trust_score", dtype=DataType.FLOAT64, description="Device fingerprint trust index (0.0 to 1.0)"),
            Feature(name="is_vpn", dtype=DataType.BOOLEAN, description="Whether traffic originated from VPN/Tor exit node"),
            Feature(name="velocity_ratio", dtype=DataType.FLOAT64, description="Velocity ratio (10m / 1h)"),
            Feature(name="composite_risk", dtype=DataType.FLOAT64, description="Trust-adjusted composite risk"),
        ],
        ttl_seconds=86400,  # 24 hours
        online=True,
        offline=True,
        transformation_name="fraud_ratios",
        description="Real-time and historical fraud prevention feature signals",
    )
    store.register_feature_view(user_features)

    # View 2: Merchant Risk Features
    merchant_features = FeatureView(
        name="merchant_risk_features",
        entity_name="merchant",
        features=[
            Feature(name="chargeback_rate", dtype=DataType.FLOAT64, description="Chargeback rate over last 90 days"),
            Feature(name="volume_30d", dtype=DataType.FLOAT64, description="Total processed volume in last 30 days"),
            Feature(name="risk_tier", dtype=DataType.STRING, description="Merchant underwriting tier (LOW, MED, HIGH)"),
            Feature(name="high_risk_country", dtype=DataType.BOOLEAN, description="Whether merchant registered in high-risk jurisdiction"),
            Feature(name="exposure_score", dtype=DataType.FLOAT64, description="Volume-weighted exposure index"),
        ],
        ttl_seconds=604800,  # 7 days
        online=True,
        offline=True,
        transformation_name="merchant_exposure",
        description="Underwriting and transactional merchant risk indicators",
    )
    store.register_feature_view(merchant_features)

    # View 3: Device Fingerprint Features
    device_features = FeatureView(
        name="device_fingerprint_features",
        entity_name="device",
        features=[
            Feature(name="browser_anomaly_score", dtype=DataType.FLOAT64, description="User-agent anomaly metric"),
            Feature(name="is_headless", dtype=DataType.BOOLEAN, description="Whether browser is Puppeteer/Selenium/Playwright"),
            Feature(name="device_age_days", dtype=DataType.INT64, description="Days since first seen device fingerprint"),
            Feature(name="login_failures_24h", dtype=DataType.INT64, description="Failed authentication attempts on device in 24h"),
        ],
        ttl_seconds=259200,  # 3 days
        online=True,
        offline=True,
        description="Hardware, browser, and network fingerprint signals",
    )
    store.register_feature_view(device_features)

    # View 4: Customer Lifetime Value Features
    clv_features = FeatureView(
        name="customer_lifetime_value_features",
        entity_name="user",
        features=[
            Feature(name="total_orders_count", dtype=DataType.INT64, description="All-time completed order count"),
            Feature(name="total_spend_amount", dtype=DataType.FLOAT64, description="All-time gross spend amount"),
            Feature(name="churn_probability", dtype=DataType.FLOAT64, description="Predicted 30-day churn probability"),
            Feature(name="avg_basket_value", dtype=DataType.FLOAT64, description="Average order value"),
        ],
        ttl_seconds=2592000,  # 30 days
        online=True,
        offline=True,
        description="Customer lifetime spend and engagement features",
    )
    store.register_feature_view(clv_features)

    return store


def seed_fraud_dataset(store: FeatureStore, num_users: int = 100, events_per_user: int = 5):
    """Generates a realistic historical and online timeline across multiple domains."""
    now = datetime.now(timezone.utc)

    # 1. User Fraud Features
    user_records: List[Dict[str, any]] = []
    for u_idx in range(1, num_users + 1):
        entity_key = f"user_{u_idx:04d}"
        base_risk = round(random.uniform(0.02, 0.35), 4)
        base_trust = round(random.uniform(0.85, 0.99), 4)

        if u_idx % 15 == 0:
            base_risk = round(random.uniform(0.65, 0.95), 4)
            base_trust = round(random.uniform(0.15, 0.45), 4)

        for step in range(events_per_user):
            offset_minutes = (events_per_user - step) * 30 + random.randint(0, 15)
            evt_time = now - timedelta(minutes=offset_minutes)

            tx_10m = random.randint(0, 4) if base_risk < 0.5 else random.randint(3, 12)
            tx_1h = tx_10m + random.randint(1, 8)
            tx_amt = round(random.uniform(15.0, 350.0) * (tx_1h / 2.0), 2)
            is_vpn = (base_risk > 0.6) and (random.random() > 0.3)

            user_records.append({
                "entity_key": entity_key,
                "event_timestamp": evt_time,
                "tx_count_10m": tx_10m,
                "tx_count_1h": tx_1h,
                "tx_amount_1h": tx_amt,
                "risk_score": base_risk,
                "device_trust_score": base_trust,
                "is_vpn": is_vpn,
            })

    df_users = pd.DataFrame(user_records)
    store.ingest_batch("user_fraud_features", df_users, sync_to_online=True)

    # 2. Merchant Risk Features
    merchant_records = []
    for m_idx in range(1, 35):
        m_key = f"merchant_{m_idx:04d}"
        cb_rate = round(random.uniform(0.002, 0.035), 4)
        vol = round(random.uniform(25000.0, 850000.0), 2)
        tier = "LOW" if cb_rate < 0.01 else ("MED" if cb_rate < 0.025 else "HIGH")
        high_risk_co = (cb_rate > 0.02) and (random.random() > 0.5)

        for step in range(3):
            offset_hours = (3 - step) * 12
            evt_time = now - timedelta(hours=offset_hours)
            merchant_records.append({
                "entity_key": m_key,
                "event_timestamp": evt_time,
                "chargeback_rate": cb_rate,
                "volume_30d": vol,
                "risk_tier": tier,
                "high_risk_country": high_risk_co,
            })
    df_merchants = pd.DataFrame(merchant_records)
    store.ingest_batch("merchant_risk_features", df_merchants, sync_to_online=True)

    # 3. Device Fingerprint Features
    device_records = []
    for d_idx in range(1, 50):
        d_key = f"device_{d_idx:04d}"
        anomaly = round(random.uniform(0.01, 0.85), 3)
        headless = anomaly > 0.7 and random.random() > 0.4
        age_days = random.randint(1, 400)
        failures = random.randint(0, 2) if anomaly < 0.5 else random.randint(3, 8)

        for step in range(3):
            offset_hours = (3 - step) * 6
            evt_time = now - timedelta(hours=offset_hours)
            device_records.append({
                "entity_key": d_key,
                "event_timestamp": evt_time,
                "browser_anomaly_score": anomaly,
                "is_headless": headless,
                "device_age_days": age_days,
                "login_failures_24h": failures,
            })
    df_devices = pd.DataFrame(device_records)
    store.ingest_batch("device_fingerprint_features", df_devices, sync_to_online=True)

    # 4. Customer Lifetime Value Features
    clv_records = []
    for u_idx in range(1, num_users + 1):
        entity_key = f"user_{u_idx:04d}"
        orders = random.randint(2, 65)
        spend = round(orders * random.uniform(35.0, 180.0), 2)
        churn = round(random.uniform(0.05, 0.65), 3)
        avg_basket = round(spend / orders, 2)

        evt_time = now - timedelta(days=1)
        clv_records.append({
            "entity_key": entity_key,
            "event_timestamp": evt_time,
            "total_orders_count": orders,
            "total_spend_amount": spend,
            "churn_probability": churn,
            "avg_basket_value": avg_basket,
        })
    df_clv = pd.DataFrame(clv_records)
    store.ingest_batch("customer_lifetime_value_features", df_clv, sync_to_online=True)

    # Warm up online inferences
    all_user_features = [
        "user_fraud_features:tx_count_10m",
        "user_fraud_features:tx_count_1h",
        "user_fraud_features:tx_amount_1h",
        "user_fraud_features:risk_score",
        "user_fraud_features:device_trust_score",
        "user_fraud_features:is_vpn",
        "user_fraud_features:velocity_ratio",
        "user_fraud_features:composite_risk",
    ]
    sample_entities = [f"user_{i:04d}" for i in range(1, min(num_users + 1, 80))]
    for ek in sample_entities:
        _ = store.get_online_features([ek], all_user_features, log_for_skew_check=True)

    return df_users


def run_demo():
    print("=" * 70)
    print(" FeatureHub — Enterprise Multi-Domain Feature Store Demo")
    print("=" * 70)

    store = FeatureStore(offline_db_path=":memory:", num_online_shards=32)
    setup_fraud_feature_store(store)
    print("1. Schema registered: user, merchant, device entities with 4 feature views.")

    print("2. Seeding multi-domain datasets...")
    df_hist = seed_fraud_dataset(store, num_users=100, events_per_user=5)
    print(f"   Ingested multi-domain historical logs into DuckDB offline store.")

    # Test Online Serving across multiple feature views
    print("\n3. Testing Online Low-Latency Multi-Domain Feature Serving:")
    user_res = store.get_online_features(
        ["user_0001", "user_0015"],
        ["user_fraud_features:tx_count_10m", "customer_lifetime_value_features:total_spend_amount"]
    )
    for r in user_res:
        print(f"   User -> {r}")

    merchant_res = store.get_online_features(
        ["merchant_0001", "merchant_0010"],
        ["merchant_risk_features:chargeback_rate", "merchant_risk_features:exposure_score"]
    )
    for r in merchant_res:
        print(f"   Merchant -> {r}")

    stats = store.online_store.read_telemetry.get_stats()
    print(f"   Online Serving Latency: p50={stats['p50_ms']}ms, p99={stats['p99_ms']}ms, QPS={stats['qps']}")

    # Test Skew Parity Verification
    print("\n4. Verifying 0% Training-Serving Skew:")
    all_features = [
        "user_fraud_features:tx_count_10m",
        "user_fraud_features:tx_count_1h",
        "user_fraud_features:tx_amount_1h",
        "user_fraud_features:risk_score",
        "user_fraud_features:device_trust_score",
        "user_fraud_features:is_vpn",
        "user_fraud_features:velocity_ratio",
        "user_fraud_features:composite_risk",
    ]
    report = store.verify_skew(all_features, sample_size=100)
    print(f"   Total Feature Checks : {report.total_checks}")
    print(f"   Parity Matches       : {report.matched_features}")
    print(f"   Mismatched Features  : {report.mismatched_features}")
    print(f"   Training-Serving Skew: {report.skew_rate_percent:.4f}%")
    print(f"   Parity Guarantee     : {report.parity_rate_percent:.4f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
