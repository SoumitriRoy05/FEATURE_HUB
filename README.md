# ⚡ FeatureHub

> **Production-grade feature store for real-time ML** — sub-millisecond online serving, point-in-time offline training data, and zero training-serving skew.

FeatureHub is a full-stack feature platform built to make machine-learning features fast, reliable, and consistent from experimentation to production. It combines an in-memory online store with an offline historical store powered by DuckDB, giving teams a single source of truth for feature definitions and transformations.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-Offline%20Store-FFF000?style=for-the-badge&logo=duckdb&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## ✨ Highlights

- ⚡ **Sub-millisecond online feature retrieval**
- 🎯 **Zero training-serving skew** through shared transformation logic
- 🦆 **DuckDB-powered offline store** with point-in-time ASOF joins
- 📡 **Real-time feature ingestion and event streaming**
- 🤖 **Live ML inference workflow**
- 📊 **Benchmark dashboard** for latency, throughput, and percentile metrics
- 🛡️ **Data quality, freshness, TTL, and drift monitoring**
- 🔄 **Backfill and offline-to-online synchronization**
- 📜 **Auditable feature activity and schema changes**
- 🧩 **Dynamic feature-view registration** with versioned metadata

## 🏗️ Architecture

```text
                 ┌─────────────────────────┐
                 │   Real-Time Ingestion   │
                 └────────────┬────────────┘
                              │
             ┌────────────────┴────────────────┐
             │                                 │
   ┌─────────▼──────────┐            ┌─────────▼──────────┐
   │   Online Store     │            │   Offline Store    │
   │ In-Memory Shards   │            │      DuckDB        │
   │ Fast ML Serving    │            │ Historical Events  │
   └─────────┬──────────┘            └─────────┬──────────┘
             │                                 │
             └────────────────┬────────────────┘
                              │
                 ┌────────────▼────────────┐
                 │ Unified Transformations │
                 │  Zero Training-Serving  │
                 │          Skew           │
                 └─────────────────────────┘
```

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/SoumitriRoy05/FEATURE_HUB.git
cd FEATURE_HUB
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run FeatureHub

```bash
python run_server.py
```

Open the dashboard at:

```text
http://127.0.0.1:8000/
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## 🖥️ Operations Hub

FeatureHub includes an interactive command center for exploring and operating the feature platform.

| Tool | Purpose |
|---|---|
| ⚡ Latency Benchmark | Measure p50, p90, p95, and p99 feature-serving latency |
| 🤖 ML Inference Engine | Run real-time prediction workflows |
| 📡 Live Stream Engine | Simulate and inspect streaming events |
| 🎯 Skew Eliminator | Verify training and serving transformation parity |
| ⏳ Time Travel / ASOF | Generate point-in-time-correct historical datasets |
| 🛡️ Data Quality Guard | Detect invalid, missing, or inconsistent feature values |
| 🔄 Backfill & Sync | Replay historical events and synchronize stores |
| 📥 Real-Time Ingestion | Ingest new feature events |
| 📊 Dataset Generator | Build model-ready training datasets |
| 🩺 Freshness & TTL | Monitor feature expiry and freshness |
| 📈 Data Drift | Track distribution shifts using PSI |
| 🧪 Online Query Console | Inspect currently served feature values |
| 📜 Audit Trail | Review feature activity and schema events |

## 📚 Feature Catalog

FeatureHub maintains a centrally managed catalog of feature views, entities, schemas, TTL policies, and transformations.

Example feature views include:

- `user_fraud_features`
- `merchant_risk_features`
- `device_fingerprint_features`
- `customer_lifetime_value_features`
- `user_session_features`

Each feature view can define:

- Target entity and join keys
- Online and offline availability
- Feature types and schema metadata
- TTL / freshness rules
- Shared transformation functions
- Versioning and audit history

## 🎯 Why Zero Skew Matters

Training-serving skew occurs when the logic used to create training data differs from the logic used in production. This causes model quality to degrade after deployment.

FeatureHub avoids this by applying the same transformations to both historical and live data:

```text
Historical events ──► Shared transformations ──► Training dataset
Live events       ──► Shared transformations ──► Online feature serving
```

The result: reliable models, reproducible features, and consistent production predictions.

## 🗂️ Project Structure

```text
FEATURE_HUB/
├── featurehub/
│   ├── server/           # API server and application logic
│   ├── offline/          # DuckDB-backed offline feature store
│   ├── online/           # Online in-memory feature store
│   ├── ui/               # Web dashboard assets
│   ├── registry.py       # Feature registry
│   ├── metadata.py       # Feature metadata and schemas
│   ├── transformations.py# Shared feature transformations
│   └── skew.py           # Training-serving parity checks
├── tests/                # Automated tests
├── examples/             # Example workflows
├── requirements.txt      # Python dependencies
└── run_server.py         # Application entry point
```

## 🧪 Use Case: Real-Time Fraud Detection

FeatureHub is ideal for high-speed fraud prevention workflows:

1. Ingest payment and device events in real time.
2. Calculate transaction velocity, device trust, and merchant risk features.
3. Serve the latest features to a fraud model in milliseconds.
4. Build historically accurate training data with ASOF joins.
5. Guarantee the model sees the same feature logic during training and production.

## 🛣️ Roadmap

- [ ] Persistent online-store adapters
- [ ] Authentication and role-based access control
- [ ] Feature version comparison
- [ ] Scheduled materialization jobs
- [ ] Model registry integration
- [ ] Docker deployment support
- [ ] Cloud object-storage support
- [ ] Expanded observability dashboards

## 🤝 Contributing

Contributions, feature ideas, and bug reports are welcome.

1. Fork this repository.
2. Create a feature branch.
3. Make your changes.
4. Add or update tests where appropriate.
5. Open a pull request.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

Built with ⚡ for fast, reliable, production-ready machine learning.
