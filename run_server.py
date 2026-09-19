"""FeatureHub Server Runner.

Seeds the FeatureStore with realistic fraud detection streaming and historical data,
and starts the FastAPI server hosting both the low-latency API and the Cyber Dashboard.
"""

import sys
import uvicorn
from featurehub.server.app import app, store
from examples.fraud_detection_demo import setup_fraud_feature_store, seed_fraud_dataset


def initialize_and_seed():
    print("[FeatureHub] Initializing FeatureStore schema and transformations...")
    setup_fraud_feature_store(store)
    print("[FeatureHub] Seeding realistic fraud detection dataset (100 users, historical timelines)...")
    seed_fraud_dataset(store, num_users=100, events_per_user=5)
    print("[FeatureHub] FeatureStore ready. Sub-millisecond p99 engine active.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    initialize_and_seed()
    host = "127.0.0.1"
    port = 8000
    print(f"\n[FeatureHub] Command Center: http://{host}:{port}")
    print(f"[FeatureHub] Interactive API Docs: http://{host}:{port}/docs\n")
    uvicorn.run(app, host=host, port=port, log_level="info")
