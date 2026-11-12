# LedgerMind AI 🧠💳

**Intelligent Financial Analytics, Fraud Detection & Event-Sourced Ledger Platform**

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-336791.svg?logo=postgresql)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker)](https://www.docker.com/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Core Engineering & Financial Principles](#-core-engineering--financial-principles)
- [Quick Start Guide](#-quick-start-guide)
  - [Prerequisites](#prerequisites)
  - [Local Environment Setup](#local-environment-setup)
  - [Environment Variables](#environment-variables)
  - [Running the Application](#running-the-application)
  - [Docker Compose Deployment](#docker-compose-deployment)
- [API Reference](#-api-reference)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Development Roadmap](#-development-roadmap)
- [Contributing & Code Standards](#-contributing--code-standards)
- [License](#-license)

---

## 🌟 Overview

**LedgerMind AI** is an enterprise-grade financial intelligence platform bridging **Data Science, Machine Learning, AI Engineering, and Systems Architecture** into a single unified system.

Financial institutions handle millions of high-velocity transactions every day. Disparate, fragmented legacy systems create severe blind spots: delayed fraud detection, error-prone reconciliations, manual financial reporting, and complex audit trails. 

LedgerMind AI solves these challenges by combining:
1. **Immutable Double-Entry Accounting** powered by an event-sourced ledger.
2. **Real-Time Fraud Detection** driven by machine learning pipelines and anomaly detection.
3. **Advanced SQL & Time-Series Financial Analytics** for actionable cash-flow intelligence.
4. **AI-Powered Financial Analyst** leveraging Retrieval-Augmented Generation (RAG) and tool calling to explain trends and audit records with zero hallucinations.

---

## 🏗️ System Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Frontend Presentation Tier               │
                    │      (Streamlit Operations Cockpit & Analyst UX)       │
                    └───────────────────────────┬────────────────────────────┘
                                                │ REST + JWT Bearer Auth
                                                ▼
     ┌──────────────────────────────────────────────────────────────────────────────────┐
     │                     FastAPI High-Performance Async Backend                        │
     │      (Ingestion Gateway, RBAC Security, Lifespan Preloading, Idempotency)        │
     └───┬──────────────────────┬───────────────────────┬───────────────────────────┬───┘
         │                      │                       │                           │
         ▼                      ▼                       ▼                           ▼
┌──────────────────┐   ┌──────────────────┐   ┌───────────────────┐       ┌─────────────────┐
│ Event-Sourced    │   │ Sub-Millisecond  │   │  Real-Time ML &   │       │  Compliance &   │
│ Double-Entry DB  │   │ Velocity Engine  │   │  TreeSHAP Pipeline│       │  SAR Generator  │
├──────────────────┤   ├──────────────────┤   ├───────────────────┤       ├─────────────────┤
│ • SELECT FOR UPD │   │ • Sliding Window │   │ • Cost Optimizer  │       │ • RBAC Enforce  │
│ • Zero-Drift Leg │   │ • 1m / 1h Counts │   │ • Top 3 SHAP Att. │       │ • PII Masking   │
│ • BIGINT Minor   │   │ • Redis 7 Fallbk │   │ • Haversine Speed │       │ • Immutable Log │
└────────┬─────────┘   └────────┬─────────┘   └─────────┬─────────┘       └────────┬────────┘
         │                      │                       │                          │
         ▼                      ▼                       ▼                          ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Data & Storage Tier (PostgreSQL 17 + Redis 7)                   │
│  ┌────────────────────────┐  ┌───────────────────────────┐  ┌───────────────────────┐  │
│  │ Domain Event Store     │  │ Balance Projections       │  │ Redis Sliding-Window  │  │
│  │ (Append-Only Journal)  │  │ (Row-Level Locking)       │  │ (Velocity Ring Buffer)│  │
│  └────────────────────────┘  └───────────────────────────┘  └───────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Key Features

| Module | Status | Description |
|:---|:---:|:---|
| 💳 **Transaction Management** | 🟢 *Active* | Complete account lifecycle, deposit/withdrawal validation, and atomic transfer processing. |
| 📖 **Event-Sourced Ledger** | 🟢 *Active* | Strict double-entry bookkeeping (`Σ Debits == Σ Credits`), immutable append-only domain event store, and rebuildable balance projections. |
| 📊 **Financial Analytics** | 🟢 *Active* | High-performance SQL aggregations, rolling velocity windows, liquidity metrics, and platform-wide volume profiling. |
| 🛡️ **ML Fraud Detection** | 🟢 *Active* | Class-weighted supervised classification, Isolation Forest anomaly scoring, and cost-calibrated decision threshold optimization. |
| 🤖 **AI Assistant & RAG** | 🟢 *Active* | Financial intelligence assistant grounded strictly in verified audit logs and database queries with zero hallucinations. |
| 📈 **Interactive Dashboard** | 🟢 *Active* | Streamlit web interface featuring real-time risk triage, transaction distributions, and ledger audit tools. |
| ⚡ **FastAPI REST API** | 🟢 *Active* | Asynchronous high-throughput endpoints, strict Pydantic v2 schemas, idempotency replay headers, JWT authentication, and interactive OpenAPI documentation. |
| 🧪 **Test & DevOps Suite** | 🟢 *Active* | 53-test automated test suite (Pytest), Docker multi-stage builds, Docker Compose, Alembic migrations, GitHub Actions CI, and 100% clean Ruff/Mypy validation. |

---

## 💻 Technology Stack

### Backend & API
- **Python 3.12+**: Modern Python with strict type hinting (`mypy`) and PEP 563/695 standards.
- **FastAPI**: Asynchronous web framework for high-throughput, low-latency REST endpoints.
- **Uvicorn**: Lightning-fast ASGI web server implementation.
- **Pydantic v2 & Pydantic-Settings**: High-performance runtime validation and environment management.

### Persistence & Data Layer
- **PostgreSQL 17**: Relational database supporting ACID guarantees, JSONB events, and window functions.
- **SQLAlchemy 2.0 (Async)**: Modern asyncio ORM and SQL Expression Language.
- **asyncpg**: High-performance asynchronous PostgreSQL driver for Python.
- **Alembic**: Environment-controlled, declarative database schema migrations.

### Data Science & Machine Learning
- **Pandas & NumPy**: Optimized in-memory data transformation, statistical profiling, and vector calculations.
- **Scikit-learn**: Feature scaling, imbalance mitigation, model selection (Random Forest, Logistic Regression, XGBoost/LightGBM).
- **PyTorch**: Deep anomaly detection and neural representations for complex fraud topologies.
- **Matplotlib & Seaborn**: Exploratory data analysis (EDA), fraud correlation heatmaps, and residual diagnostics.
- **Joblib**: Deterministic serialization for machine learning pipelines and transformers.

### AI & LLM Systems
- **OpenAI API / Local LLMs**: Controlled tool calling, synthetic statement generation, and natural language query execution.
- **RAG Architecture**: Retrieval-augmented generation for auditing rules, financial policies, and report verification.

### DevOps & Code Quality
- **Docker & Docker Compose**: Multi-stage containerization with rootless runtime security.
- **Pytest, pytest-asyncio, pytest-cov**: Comprehensive unit, integration, and ML invariant test coverage.
- **Ruff & Mypy**: Ultra-fast linting, formatting, and static typing enforcement.

---

## 📁 Project Structure

```
ledgermind-ai/
├── app/                              # Application source code
│   ├── accounts/                     # Account management and lifecycle
│   ├── ai_assistant/                 # LLM integration, prompt templates, and RAG
│   ├── analytics/                    # Financial metrics, SQL queries, and aggregations
│   ├── core/                         # Core infrastructure
│   │   ├── config.py                 # Pydantic Settings and env loader
│   │   ├── database.py               # Async SQLAlchemy engine and session factory
│   │   └── security.py               # Authentication and authorization guards
│   ├── fraud/                        # Fraud prediction and inference services
│   ├── ledger/                       # Event-sourced double-entry accounting engine
│   ├── transactions/                 # Transaction recording and routing
│   └── main.py                       # FastAPI application factory and lifespan hooks
├── ml/                               # Machine Learning workspace
│   ├── data/                         # Datasets (git-ignored raw data, references)
│   ├── notebooks/                    # Jupyter notebooks for EDA and model experiments
│   ├── preprocessing/                # Feature extractors, scalers, and pipelines
│   ├── training/                     # Reproducible model training scripts
│   └── models/                       # Versioned model artifacts (.joblib / .pt)
├── tests/                            # Test suite
│   ├── conftest.py                   # Shared fixtures and async test client
│   ├── unit/                         # Unit tests (zero external network/db dependencies)
│   ├── integration/                  # Database and multi-service integration tests
│   ├── api/                          # End-to-end API endpoint tests
│   └── ml/                           # ML accuracy, data drift, and pipeline tests
├── migrations/                       # Alembic database migration revisions
├── docs/                             # Architecture diagrams, specifications, and notes
├── scripts/                          # Seeding, maintenance, and utility scripts
├── .env.example                      # Template for environment configuration
├── .gitignore                        # Git exclusion rules
├── alembic.ini                       # Alembic migration configuration
├── docker-compose.yml                # Multi-container orchestration (App + Postgres)
├── Dockerfile                        # Multi-stage container build
├── pyproject.toml                    # Declarative build system, dependencies, and tools
└── README.md                         # Project documentation
```

---

## 🛡️ Core Engineering & Financial Principles

1. **Exact Monetary Representation (No Floating-Point Math)**:
   - Floating-point representations (e.g. `0.1 + 0.2 != 0.3`) cause disastrous rounding errors in financial ledgers.
   - All monetary values are represented as **integer minor units** (e.g., cents: `$100.50` is stored as `10050`).
2. **Immutable Audit Trails (Event Sourcing)**:
   - Financial transactions cannot be modified or deleted (`UPDATE`/`DELETE` are restricted on historical ledgers).
   - All balance changes are represented by balanced double-entry debits and credits (`Σ Debits == Σ Credits`).
3. **Idempotent Financial Operations**:
   - Every state-changing transaction accepts a unique `Idempotency-Key` header, guaranteeing that network retries never lead to duplicate charges or transfers.
4. **Fraud Evaluation Beyond Accuracy**:
   - Because fraud datasets feature extreme class imbalance (often < 1% positive class), standard accuracy is a deceptive metric. Models are evaluated using **Precision-Recall AUC**, **Recall at fixed Precision**, and **Cost-sensitive Cost Matrices**.
5. **Grounded AI Intelligence**:
   - AI assistant tools operate strictly through parameterized, read-only tools with verified role-based access control, preventing prompt injections and unauthorized data exposure.

---

## 🚀 Quick Start Guide

### Prerequisites

- **Python**: Version `3.12` or higher
- **Docker & Docker Compose**: For containerized database and services
- **Git**: Version control

---

### Local Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/ledgermind-ai.git
   cd ledgermind-ai
   ```

2. **Create and activate a virtual environment:**
   - **On Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
     *(If PowerShell blocks script execution, run: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`)*
   - **On macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   # Install base packages + dev dependencies in editable mode
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"
   
   # Or install all extras (including ML, AI, and Dashboard)
   python -m pip install -e ".[all]"
   ```

---

### Environment Variables

Copy the provided template and customize your local settings:

```bash
cp .env.example .env
```

Key environment configurations (`.env`):
```ini
# Application
APP_NAME="LedgerMind AI"
APP_ENV=development
DEBUG=True
LOG_LEVEL=INFO

# PostgreSQL Connection
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=ledgermind
POSTGRES_PASSWORD=changeme
POSTGRES_DB=ledgermind_db

# Security
SECRET_KEY=generate-a-secure-random-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# ML Inference
MODEL_PATH=ml/models/fraud_model.joblib
PREDICTION_THRESHOLD=0.5
```

---

### Running the Application

1. **Start PostgreSQL via Docker Compose:**
   ```bash
   docker compose up -d db
   ```

2. **Apply Database Migrations:**
   ```bash
   alembic upgrade head
   ```

3. **Launch the FastAPI Server:**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access the application:**
   - **Interactive API Documentation (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Alternative API Reference (ReDoc):** [http://localhost:8000/redoc](http://localhost:8000/redoc)
   - **Health Check Endpoint:** [http://localhost:8000/health](http://localhost:8000/health)

5. **Launch the Production Operations Cockpit (Streamlit Dashboard):**
   ```bash
   streamlit run app/dashboard.py
   ```
   Explore the 7 institutional operations modules at [http://localhost:8501](http://localhost:8501):
   - 🚨 **Incident Command & Live Fraud Triage Queue**: Real-world FinOps/SOC alert stream with anomaly badges and 1-click Account Freeze, FIDO2 Step-up 2FA, and Investigation workflows.
   - 🔬 **Deep Transaction Investigator & SHAP XAI**: Interactive PCA vector analysis with TreeExplainer attributions isolating exact risk forces.
   - 🕸️ **AML & Mule Network Flow Visualizer**: Graph node architecture tracking structuring (smurfing below €10k) and pass-through layering across mule clusters.
   - 📖 **Ledger & Cryptographic Proof-of-Solvency**: Live double-entry bookkeeping with SHA-256 block hash chaining and strict zero-drift audit ($\Sigma\text{Debits} == \Sigma\text{Credits}$).
   - ⚡ **Autonomous Cyber Attack Simulator**: High-velocity stress testing simulating 250-txn botnet card bursts, smurfing surges, and business peaks at sub-5ms SLA.
   - 🎯 **Cost Frontier & Threshold Optimizer**: Interactive financial loss curve balancing user friction costs against fraud chargeback losses to find optimal cutoff $\tau^*$.
   - 🤖 **AI Copilot & Regulatory SAR Generator**: Natural language compliance analyst with one-click formal FinCEN Suspicious Activity Report (SAR) generation and SQL audit evidence drawers.

---

### Docker Compose Deployment

To build and run the complete ecosystem (PostgreSQL database and FastAPI application) in isolated containers:

```bash
# Build and start services in the background
docker compose up -d --build

# Monitor live application logs
docker compose logs -f app

# Tear down containers and preserve database volume
docker compose down
```

---

## 📡 API Reference (v1.0 Production)

All core business endpoints are versioned under `/api/v1` with Kubernetes liveness and deep readiness probes at root.

| Method | Endpoint | Description | Auth / RBAC | Status |
|---|---|---|---|---|
| `GET` | `/healthz` | Kubernetes liveness probe (HTTP 200 process check) | Public | 🟢 Active |
| `GET` | `/readyz` | Kubernetes deep readiness (DB ping + Redis + Model state) | Public | 🟢 Active |
| `GET` | `/health` | Deep platform health check and component telemetry | Public | 🟢 Active |
| `POST` | `/api/v1/auth/token` | Issue signed JWT access token with user role | Public | 🟢 Active |
| `POST` | `/api/v1/accounts` | Create financial account and initialize projection | Bearer JWT | 🟢 Active |
| `GET` | `/api/v1/accounts/{id}` | Retrieve verified balance projection (BIGINT minor units) | Bearer JWT | 🟢 Active |
| `GET` | `/api/v1/accounts/{id}/transactions` | Paginated journal entries (debits/credits) | Bearer JWT | 🟢 Active |
| `POST` | `/api/v1/transfers` | Atomic double-entry transfer (`SELECT FOR UPDATE`) | Bearer JWT | 🟢 Active |
| `POST` | `/api/v1/transactions/ingest` | Real-time webhook ingestion with `Idempotency-Key` | Public / Gateway | 🟢 Active |
| `POST` | `/api/v1/fraud/predict` | ML scoring & TreeSHAP localized attributions | Bearer JWT | 🟢 Active |
| `GET` | `/api/v1/fraud/model-info` | Active ensemble metadata and optimal threshold | Bearer JWT | 🟢 Active |
| `GET` | `/api/v1/analytics/summary` | Real-time platform liquidity and transfer volume | Bearer JWT | 🟢 Active |
| `POST` | `/api/v1/compliance/freeze` | Freeze compromised account (compensating event) | `Compliance_Officer` / `Admin` | 🟢 Active |
| `POST` | `/api/v1/compliance/step-up-2fa` | Trigger out-of-band FIDO2 / 2FA challenge | `Analyst`+ | 🟢 Active |
| `POST` | `/api/v1/compliance/sar-generate` | Generate FinCEN SAR narrative with PII redaction | `Compliance_Officer` / `Admin` | 🟢 Active |
| `GET` | `/api/v1/compliance/audit-logs` | Retrieve immutable operator compliance audit trail | `Compliance_Officer` / `Admin` | 🟢 Active |
| `POST` | `/api/v1/ai/ask` | Natural language grounded queries handled by AI | Bearer JWT | 🟢 Active |

---

## 🚀 Quick Start & Production Workflows

```bash
# 1. Train production calibrated ensemble model (saves ml/models/fraud_model.joblib)
python ml/train.py

# 2. Start the high-performance FastAPI backend server
uvicorn backend.main:app --reload --port 8000

# 3. Stream real-world continuous traffic (Legitimate, Account Takeover, Smurfing)
python scripts/live_traffic_generator.py --rate 1.0

# 4. Open the interactive operations dashboard (Presentation Tier)
streamlit run frontend/dashboard.py
```

---

## 🖥️ Interactive Dashboard

LedgerMind AI features an interactive Streamlit operations dashboard for real-time monitoring and investigation:

```bash
# Launch the dashboard
python -m streamlit run app/dashboard.py
```

### Dashboard Capabilities:
- **Real-Time Fraud Triage**: Live transaction scoring, SHAP explainability attributions, risk distributions, and flagged alerts.
- **Settlement & Liquidity Analytics**: Visual breakdown of total transaction volume and cash movement.
- **Event-Sourced Ledger Explorer**: Inspect the immutable journal transaction log and balance projections.
- **AI Financial Assistant**: Ask grounded natural language questions directly against the authoritative ledger.
- **Model Performance Benchmarks**: ROC curves, Precision-Recall curves, and cost matrix comparison.

---

## 🧪 Testing & Quality Assurance

The codebase adheres to strict testing practices with markers for test isolation:

```bash
# Run the complete test suite with coverage
pytest -v --cov=app --cov=ml

# Run fast unit tests only (no external database required)
pytest -m unit

# Run database integration tests
pytest -m integration

# Run machine learning pipeline tests
pytest -m ml
```

### Static Analysis & Formatting

```bash
# Lint code with Ruff
ruff check .

# Format code with Ruff
ruff format .

# Static type verification with Mypy
mypy --explicit-package-bases app ml scripts tests
```

---

## 🗺️ Development Roadmap

- [x] **Phase 0: Planning & Architecture** — Financial domain model, ledger invariants, and system architecture design.
- [x] **Phase 1: Environment & Project Setup** — Hatchling build config, virtual environment, dependencies, Ruff/Mypy setup.
- [x] **Phase 2: Dataset Acquisition & Validation** — ULB / Worldline benchmark acquisition, validation schema, and synthetic generator.
- [x] **Phase 3: Exploratory Data Analysis (EDA)** — Headless Matplotlib/Seaborn statistical profiling, percentiles, and publication-ready figures.
- [x] **Phase 4: Feature Engineering Pipeline** — Chronological train/val/test splitting, cyclical time transformer, RobustScaler, and scikit-learn pipeline serialization.
- [x] **Phase 5: Baseline Supervised Fraud Model** — Class-weighted Logistic Regression, confusion matrix, precision, recall, F1, PR-AUC, and ROC-AUC.
- [x] **Phase 6: Model Comparison & Anomaly Detection** — Random Forest, Isolation Forest anomaly scoring, and cost-calibrated decision threshold optimization.
- [x] **Phase 7: Financial Domain Model** — Exact integer minor units (`Money`), immutable journal entries, and balanced double-entry invariants (`Σ Debits == Σ Credits`).
- [x] **Phase 8: PostgreSQL Schema & Relational Models** — SQLAlchemy 2.0 async models for accounts, transactions, entries, projections, and idempotency.
- [x] **Phase 9: Event History & Balance Projections** — Append-only domain event log, atomic transfer transactions, and balance projection replay.
- [x] **Phase 10: Idempotent Transfer API** — FastAPI `/transfers` endpoint with `Idempotency-Key` deduplication, optimistic validation, and atomic commits.
- [x] **Phase 11: Real-Time Fraud Inference Endpoint** — Low-latency `/fraud/predict` and `/fraud/model-info` with model caching in FastAPI lifespan.
- [x] **Phase 12: Financial Analytics Engine** — Platform-wide SQL aggregations via `/analytics/summary` for turnover, balances, and volume profiling.
- [x] **Phase 13: AI Financial Assistant** — Grounded `/ai/ask` endpoint executing read-only database queries to eliminate hallucinations.
- [x] **Phase 14: Interactive Operations Dashboard** — Streamlit operations visualizer with 5 specialized tabs for triage, analytics, ledger, AI, and benchmarks.
- [x] **Phase 15: Security & Threat Modeling** — STRIDE threat model matrix, financial integrity invariants, and security controls (`docs/threat_model.md`).
- [x] **Phase 16: CI/CD Automation** — GitHub Actions workflow running Ruff, Mypy, and Pytest with coverage verification (`.github/workflows/ci.yml`).
- [x] **Phase 17: Containerization & Database Migrations** — Multi-stage production `Dockerfile`, `docker-compose.yml`, and initial Alembic database migration.
- [x] **Phase 18: System Verification & Portfolio Finalization** — End-to-end test suite passing (45 tests, >81% coverage), zero lint/type errors, and complete architecture documentation.

---

## 🤝 Contributing & Code Standards

1. Ensure all code conforms to **PEP 8** via `ruff`.
2. All functions must include complete **Type Hints** and satisfy `mypy` strict type checking.
3. Every financial calculation must use minor units or exact integer representation.
4. All new features must include corresponding **Pytest** tests with >= 80% coverage.

---

## 📄 License

This project is open-source and licensed under the [MIT License](LICENSE).
