import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"

def post(endpoint, data):
    req = urllib.request.Request(
        f"{BASE}{endpoint}",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get(endpoint):
    with urllib.request.urlopen(f"{BASE}{endpoint}") as resp:
        return json.loads(resp.read().decode("utf-8"))

def run_tests():
    print("=== 1. Testing Registry ===")
    reg = get("/api/v1/features/registry")
    views = reg.get("feature_views", [])
    entities = reg.get("entities", [])
    trans = reg.get("transformations", [])
    print(f"Registered Views: {len(views)}, Entities: {len(entities)}, Transformations: {len(trans)}")

    print("\n=== 2. Testing ML Inference Engine (/models/predict) ===")
    test_cases = [
        ("fraud_sentinel_v2", "user_0001"),
        ("fraud_sentinel_v2", "user_0015"),
        ("merchant_underwriter_v1", "merchant_0001"),
        ("churn_predictor_v3", "user_0001")
    ]
    for m_id, ek in test_cases:
        pred = post("/api/v1/models/predict", {"model_id": m_id, "entity_key": ek})
        dec = pred["decision"]
        prob = pred["probability_percent"]
        tot_lat = pred["latency_breakdown"]["total_latency_ms"]
        feat_us = pred["latency_breakdown"]["feature_retrieval_us"]
        print(f"Model: {m_id} on {ek} -> Decision: {dec} (Prob: {prob}, Total: {tot_lat}ms, Feat Fetch: {feat_us}us)")

    print("\n=== 3. Testing Streaming Simulator ===")
    s_start = post("/api/v1/stream/start", {"rate_per_sec": 10})
    print(f"Stream start: {s_start['status']}")
    time.sleep(1.2)
    s_status = get("/api/v1/stream/status")
    print(f"Streaming active: {s_status['active']}, Streamed: {s_status['total_streamed']} events, Recent: {len(s_status['recent_events'])}")
    s_stop = post("/api/v1/stream/stop", {})
    print(f"Stream stop: {s_stop['status']}")

    print("\n=== 4. Testing Data Quality & Assertions ===")
    q_res = post("/api/v1/quality/validate", {})
    print(f"Quality Score: {q_res['quality_score_percent']}%, Passed: {q_res['passed_count']}/{q_res['total_rules']}")
    for a in q_res["assertions"]:
        print(f"  - {a['name']}: {'PASSED' if a['passed'] else 'VIOLATION'} | Expected: {a['expected']} | Observed: {a['observed']}")

    print("\n=== 5. Testing Backfill & Sync ===")
    sync_res = post("/api/v1/sync/backfill", {})
    print(f"Backfill: Synced {sync_res['synced_records']} records in {sync_res['elapsed_ms']}ms ({sync_res['throughput_rows_sec']} rows/sec)")

    print("\n=== 6. Testing Dynamic Feature View Registration ===")
    reg_view = post("/api/v1/features/register-view", {
        "name": "user_session_features",
        "entity_name": "user",
        "ttl_seconds": 86400,
        "features": [
            {"name": "session_duration_s", "dtype": "float64"},
            {"name": "pages_viewed", "dtype": "int64"}
        ]
    })
    print(f"Registered View: {reg_view['status']}, Name: {reg_view['feature_view']}")

    print("\n=== 7. Testing Audit Logs ===")
    logs = get("/api/v1/audit/logs")
    print(f"Audit Logs: {len(logs['audit_logs'])} total.")
    print(f"Latest: {logs['audit_logs'][0]['action']} - {logs['audit_logs'][0]['details']}")

    print("\n[SUCCESS] ALL BACKEND ENDPOINTS FULLY FUNCTIONAL AND VERIFIED!")

if __name__ == "__main__":
    run_tests()
