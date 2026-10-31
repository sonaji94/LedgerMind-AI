# LedgerMind AI

**AI-Powered Financial Analytics, Fraud Detection, and Intelligent Reporting Platform**

---

## Overview

LedgerMind AI is an end-to-end financial intelligence platform that combines Data Science, Machine Learning, AI Engineering, and Backend Development into a single, well-tested system.

The platform records financial transactions, analyzes financial data, detects potentially suspicious transaction patterns using machine learning, and provides an AI-powered assistant for financial analytics and reporting.

## Features

| Module | Description | Status |
|--------|-------------|--------|
| 🏦 **Transaction Management** | Create accounts, record and retrieve transactions | 🔲 Planned |
| 📒 **Event-Sourced Ledger** | Double-entry accounting with append-only event store | 🔲 Planned |
| 📊 **Data Analytics** | SQL analytics, trend analysis, financial summaries | 🔲 Planned |
| 🔍 **Fraud Detection** | ML-based fraud risk scoring and anomaly detection | 🔲 Planned |
| 🤖 **AI Assistant** | LLM-powered financial Q&A with RAG | 🔲 Planned |
| 📈 **Dashboard** | Interactive Streamlit analytics dashboard | 🔲 Planned |
| 🌐 **REST API** | FastAPI endpoints for all functionality | 🔲 Planned |
| ✅ **Testing & CI/CD** | Pytest, Docker, GitHub Actions | 🔲 Planned |

## Tech Stack

- **Language:** Python 3.15
- **API:** FastAPI + Uvicorn
- **Database:** PostgreSQL + SQLAlchemy (async) + Alembic
- **Data Science:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, PyTorch
- **AI/LLM:** OpenAI, RAG, Embeddings
- **Dashboard:** Streamlit
- **Testing:** Pytest (unit, integration, API, ML)
- **Deployment:** Docker, Docker Compose, GitHub Actions

## Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL 17+ (or use Docker)
- Git

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd ledgermind-ai

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your database credentials

# Start PostgreSQL (via Docker)
docker compose up -d db

# Run database migrations
alembic upgrade head

# Start the development server
uvicorn app.main:app --reload
```

### Using Docker Compose

```bash
# Start everything (database + app)
docker compose up -d

# View logs
docker compose logs -f app

# Stop
docker compose down
```

### Run Tests

```bash
# All tests
pytest

# Unit tests only
pytest -m unit

# With coverage
pytest --cov=app --cov-report=html
```

## API Documentation

Once the server is running:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

## Project Structure

```
ledgermind-ai/
├── app/                        # Application source code
│   ├── core/                   # Config, database, shared utilities
│   ├── accounts/               # Account management
│   ├── ledger/                 # Event-sourced double-entry ledger
│   ├── transactions/           # Transaction recording
│   ├── analytics/              # SQL analytics and reporting
│   ├── fraud/                  # ML fraud detection
│   ├── ai_assistant/           # LLM-powered assistant
│   └── main.py                 # FastAPI application factory
├── ml/                         # Machine learning workspace
│   ├── data/                   # Datasets (raw data git-ignored)
│   ├── notebooks/              # Jupyter notebooks for EDA
│   ├── preprocessing/          # Data cleaning and features
│   ├── training/               # Model training scripts
│   └── models/                 # Serialized model artifacts
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests (need DB)
│   ├── api/                    # API endpoint tests
│   └── ml/                     # ML pipeline tests
├── migrations/                 # Alembic database migrations
├── docs/                       # Project documentation
├── scripts/                    # Utility scripts
├── pyproject.toml              # Dependencies and tool config
├── Dockerfile                  # Multi-stage production build
├── docker-compose.yml          # PostgreSQL + App services
├── alembic.ini                 # Alembic migration config
└── .env.example                # Environment variable template
```

## Development Roadmap

See [PROJECT NOTES](docs/) for the full 16-phase development plan.

## License

MIT
