# PriceScope

[![CI](https://github.com/Technocrat-dev/Marketplace-Price-Prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Technocrat-dev/Marketplace-Price-Prediction/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade multimodal deep learning system that predicts marketplace product prices from text, categorical features, and item metadata. Built end-to-end: from raw data ingestion and model training to a served REST API and interactive frontend.

Trained on **1.48 million** Mercari product listings, achieving **0.420 RMSLE** on held-out test data with a model that fuses bidirectional LSTMs (with optional self-attention) for text understanding and learned categorical embeddings for structured features.

Beyond the core model, the project includes a **fine-tuned DistilBERT comparison** (same splits and objective, so classic-vs-transformer trade-offs are measured, not assumed — the transformer landed within ~2% of the BiLSTM at 4.5x the parameters, so the lighter model stays in production) and an **LLM-powered listing analysis** endpoint (Gemini or Claude, pluggable) that pairs the ML price estimate with an LLM's independent estimate and a structured listing critique.

---

## Table of Contents

- [Screenshots](#screenshots)
- [Architecture](#architecture)
- [Results](#results)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [ML Engineering](#ml-engineering)
- [Configuration](#configuration)
- [Testing](#testing)
- [License](#license)

---

## Screenshots

**Prediction Form** -- Enter product details and get an instant price estimate with confidence range.

![Prediction Form](docs/screenshots/3c93d1bc-2c3f-43d7-962b-a2a2b91b9342.png)

**Model Dashboard** -- Live training metrics, loss curves, and recent prediction history.

![Model Dashboard](docs/screenshots/3cabb305-0ab8-4d78-927f-d4005a7e7537.png)

**Explore Products** -- Browse the catalog with category/brand analytics and full-text search.

![Explore Products](docs/screenshots/ce1e6922-4479-4a48-90c8-2d3a211e632e.png)

---

## Architecture

The model jointly encodes product names, descriptions, and structured metadata through three parallel branches, then fuses them for price regression.

```
Input Layer
------------------------------------------------------------
  Product Name          Description           Structured Features
  (tokenized)           (tokenized)           brand, category (x3),
                                              condition, shipping
       |                     |                       |
       v                     v                       v
  +-----------+        +-----------+          +-------------+
  |  Embedding |        |  Embedding |          | 5 Embedding |
  |  (64-dim)  |        |  (64-dim)  |          |  Layers     |
  +-----------+        +-----------+          +------+------+
       |                     |                       |
       v                     v                       v
  +-----------+        +-----------+          +-------------+
  |  BiLSTM   |        |  BiLSTM   |          |    Dense    |
  |  (128-d)  |        |  (128-d)  |          |   + BN +    |
  |  2 layers |        |  2 layers |          |   Dropout   |
  +-----------+        +-----------+          +------+------+
       |                     |                       |
       v                     v                       |
  [Optional]           [Optional]                    |
  Attention            Attention                     |
       |                     |                       |
       +---------------------+-----------------------+
                             |
                      Concatenation (576-d)
                             |
                             v
                    +------------------+
                    |   Fusion MLP     |
                    |  576 > 256 > 128 |
                    |  + BatchNorm     |
                    |  + Dropout (0.3) |
                    +--------+---------+
                             |
                             v
                      log1p(price)
```

**Tech Stack**

| Layer       | Technology                                                |
|-------------|-----------------------------------------------------------|
| Training    | PyTorch, Adam, RMSLE loss, ReduceLR / Cosine scheduling   |
| Tuning      | Optuna with MedianPruner                                  |
| Baselines   | XGBoost, LightGBM, Ridge                                  |
| Explainability | SHAP (TreeExplainer)                                   |
| Export      | ONNX with validation                                      |
| API         | FastAPI, Uvicorn, SlowAPI rate limiting, Pydantic v2      |
| Database    | MongoDB with indexed product and prediction repositories  |
| Frontend    | Next.js 14, TypeScript, CSS Modules, Recharts             |
| DevOps      | Docker multi-stage builds, docker-compose, GitHub Actions |

---

## Results

### Deep Learning Model

| Metric           | Test Set    |
|------------------|-------------|
| RMSLE            | **0.420** |
| MAE              | $9.33       |
| Median AE        | $4.30       |
| R-squared        | 0.561       |
| Parameters       | 14.8M       |

### Baseline Comparison

| Model            | RMSLE   | MAE      | R-squared | Train Time |
|------------------|---------|----------|-----------|------------|
| **BiLSTM + MLP** | **0.420** | **$9.33** | **0.561** | ~15 min  |
| XGBoost          | 0.555   | $12.18   | 0.301     | 8.7s       |
| LightGBM         | 0.559   | $12.34   | 0.295     | 3.2s       |
| Ridge Regression | 0.612   | $14.80   | 0.221     | 0.3s       |

The deep learning model outperforms all tabular baselines because it can extract semantic signals from free-text product names and descriptions that tree-based models cannot access.

### Transformer Comparison

A fine-tuned **DistilBERT** variant ([`scripts/train_transformer.py`](scripts/train_transformer.py)) swaps the BiLSTM text branches for a pretrained transformer over `name [SEP] description`, fused with the identical tabular encoder, splits, and RMSLE objective — so the two architectures are directly comparable. Measured results (all metrics computed identically, raw price scale — see [`outputs/transformer_results.json`](outputs/transformer_results.json)):

| Model                            | Test RMSLE | MAE     | Median AE | R-squared | Params |
|----------------------------------|------------|---------|-----------|-----------|--------|
| **BiLSTM + MLP** (11 epochs)     | **0.420**  | **$9.33** | **$4.30** | **0.561** | 14.8M |
| DistilBERT + tabular (1 epoch)   | 0.427      | $9.48   | $4.40     | 0.536     | 66.8M  |

After a single fine-tuning epoch on 1.18M listings (RTX 4050 laptop GPU), the pretrained transformer comes within ~2% of the task-tuned BiLSTM — but does not beat it, while carrying **4.5x the parameters** and roughly **10x the inference cost**. This is a sensible outcome for this domain: marketplace listing text is short, noisy, and brand-heavy, so DistilBERT's pretrained general-language knowledge buys less than it would on longer natural prose, and the BiLSTM's task-specific vocabulary and 11 epochs of training close the gap. The fine-tune was stopped after epoch 1 once validation RMSLE plateaued near-parity — the marginal gain from further epochs did not justify the compute for a comparison study.

**The engineering call:** the BiLSTM stays in production. For a high-throughput pricing API, accuracy-per-millisecond matters as much as raw RMSLE, and near-parity accuracy does not justify a 10x serving-cost increase.

```bash
# Reproduce: full fine-tune (GPU required); results land in outputs/transformer_results.json
python scripts/train_transformer.py --epochs 2

# CPU smoke test on a subsample
python scripts/train_transformer.py --sample 3000 --epochs 1
```

### Dataset

| Split      | Samples    |
|------------|------------|
| Training   | 1,185,328  |
| Validation | 148,166    |
| Test       | 148,167    |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- MongoDB (optional, for data persistence)
- [Kaggle API credentials](https://www.kaggle.com/docs/api) (for dataset download)

### 1. Clone and install

```bash
git clone https://github.com/Technocrat-dev/Marketplace-Price-Prediction.git
cd Marketplace-Price-Prediction

python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 2. Download and preprocess data

```bash
python data/download.py

python -c "from src.data.preprocess import run_preprocessing_pipeline; run_preprocessing_pipeline('config/config.yaml')"
```

### 3. Train

```bash
# Full training
python scripts/train.py

# Quick test (2 epochs)
python scripts/train.py --quick

# Resume from checkpoint
python scripts/train.py --resume outputs/checkpoints/best_model.pt
```

### 4. Serve

```bash
# Start the API
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000

# Start the frontend (separate terminal)
cd frontend && npm install && npm run dev
```

Open [http://localhost:3000](http://localhost:3000) for the web interface or [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API docs.

---

## Docker Deployment

Launch the entire stack with one command:

```bash
docker compose up --build
```

| Service      | URL                          | Description              |
|--------------|------------------------------|--------------------------|
| API          | http://localhost:8000        | FastAPI prediction server |
| Frontend     | http://localhost:3000        | Next.js web interface     |
| MongoDB      | mongodb://localhost:27017    | Data persistence          |
| API Docs     | http://localhost:8000/docs   | Swagger UI                |

---

## API Reference

Full interactive documentation available at `/docs` when the server is running.

### POST /predict -- Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Nike Air Max 90",
    "item_description": "Classic sneakers, barely worn, size 10",
    "category_name": "Men/Shoes/Athletic",
    "brand_name": "Nike",
    "item_condition_id": 2,
    "shipping": 1
  }'
```

```json
{
  "predicted_price": 68.42,
  "predicted_log_price": 4.23,
  "confidence_range": { "low": 54.74, "high": 85.53 },
  "input_summary": {
    "name": "nike air max 90",
    "brand": "nike",
    "category": "men/shoes/athletic",
    "condition": "2",
    "shipping": "seller pays"
  }
}
```

### POST /predict/batch -- Batch prediction (up to 100 items)

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [{"name": "iPhone 12 Case"}, {"name": "Vintage Dress"}]}'
```

### POST /predict/csv -- CSV upload (up to 500 rows)

```bash
curl -X POST http://localhost:8000/predict/csv -F "file=@products.csv"
```

### POST /predict/explain -- Prediction with feature analysis

```bash
curl -X POST http://localhost:8000/predict/explain \
  -H "Content-Type: application/json" \
  -d '{"name": "Nike Air Max 90", "brand_name": "Nike"}'
```

### POST /predict/analyze -- AI listing analysis (LLM)

Runs the trained model, then asks an LLM for an independent price estimate and a structured critique of the listing — quality score, strengths, concrete improvements, and an optional better title. The response compares both estimates and flags how strongly they agree.

```bash
curl -X POST http://localhost:8000/predict/analyze \
  -H "Content-Type: application/json" \
  -d '{"name": "Nike Air Max 90", "brand_name": "Nike", "item_description": "Barely worn"}'
```

```json
{
  "prediction": { "predicted_price": 22.50, "...": "..." },
  "ai_analysis": {
    "llm_estimated_price": 25.0,
    "price_reasoning": "...",
    "listing_score": 6,
    "strengths": ["Brand is specified"],
    "improvements": ["Add size information", "Describe the condition in detail"],
    "suggested_title": "Nike Air Max 90 - Men's Size 10, Good Condition"
  },
  "comparison": { "model_price": 22.5, "llm_price": 25.0, "delta_pct": 11.1, "agreement": "close" },
  "llm_model": "gemini-3.5-flash",
  "llm_provider": "gemini"
}
```

The provider is pluggable behind a shared Pydantic output schema: set `GEMINI_API_KEY` ([free tier via Google AI Studio](https://aistudio.google.com)) or `ANTHROPIC_API_KEY` in the server environment — Gemini takes precedence if both are present, and the endpoint returns 503 (with the frontend degrading gracefully) when neither is set. Models and token budget are configured under the `llm:` section of [`config/config.yaml`](config/config.yaml); both providers use schema-constrained structured outputs.

### Other endpoints

| Endpoint                | Method | Description                          |
|-------------------------|--------|--------------------------------------|
| `/model/info`           | GET    | Architecture, metrics, loss curves   |
| `/products/search?q=`   | GET    | Search product catalog               |
| `/products/stats`       | GET    | Category and brand statistics        |
| `/predictions/recent`   | GET    | Prediction history                   |
| `/health`               | GET    | Service health and status            |

### API Features

| Feature             | Detail                                            |
|---------------------|---------------------------------------------------|
| Rate limiting       | 60 req/min (predict), 10 req/min (batch, analyze) |
| Response caching    | LRU cache with TTL, keyed by MD5 of request       |
| API key auth        | Optional `X-API-Key` header (set in config)       |
| Request logging     | Method, path, latency, client IP per request      |
| Input validation    | Pydantic v2 schemas with field constraints         |

---

## Project Structure

```
Marketplace-Price-Prediction/
  config/
    config.yaml               Centralized hyperparameters and paths
  data/
    download.py               Kaggle dataset downloader
    raw/                      Raw TSV files (gitignored)
    processed/                NumPy arrays + vocabularies (gitignored)
  frontend/
    src/app/                  Pages: predict, dashboard, explore, about
    src/components/           Navbar, shared UI components
    src/lib/api.ts            Typed API client
    Dockerfile                Multi-stage Next.js build
  scripts/
    train.py                  Training entry point with checkpointing
    train_baselines.py        XGBoost, LightGBM, Ridge baselines
    train_transformer.py      DistilBERT + tabular fine-tuning comparison
    tune.py                   Optuna hyperparameter search with pruning
    explain.py                SHAP feature importance analysis
    ingest_data.py            MongoDB data ingestion
    export_onnx.py            ONNX export with output validation
  src/
    data/                     Dataset, preprocessing, feature engineering
    db/                       MongoDB client and repositories
    models/                   BiLSTM + Attention + TabularEncoder + Fusion
                              + DistilBERT transformer variant
    serving/                  FastAPI app, Pydantic schemas, middleware,
                              LLM listing analyzer (llm.py)
    training/                 Trainer loop, evaluation, metrics
  tests/                      Unit and integration tests (pytest)
  .github/workflows/ci.yml   CI: lint, test, Docker build, frontend build
  Dockerfile                  Multi-stage API container (Python 3.11-slim)
  docker-compose.yml          Full-stack orchestration (API + Frontend + MongoDB)
```

---

## ML Engineering

### Hyperparameter Tuning

```bash
python scripts/tune.py --n-trials 30 --timeout 3600
```

Uses Optuna with `MedianPruner` to search over text encoder dimensions, fusion depth, dropout rates, learning rate, and attention. Unpromising trials are terminated early to save compute.

### Baseline Comparison

```bash
python scripts/train_baselines.py
```

Trains XGBoost, LightGBM, and Ridge regression on the same train/val/test splits for a fair comparison. Results are saved to `outputs/baseline_results.json`.

### SHAP Explainability

```bash
python scripts/explain.py --sample 100
```

Generates per-feature importance scores using SHAP TreeExplainer on an XGBoost proxy model. Output saved to `outputs/shap_explanations.json`.

### ONNX Export

```bash
python scripts/export_onnx.py
```

Exports the trained model to ONNX format with dynamic batch axes. Validates that ONNX outputs match PyTorch within 1e-4 tolerance.

---

## Configuration

All settings are centralized in [`config/config.yaml`](config/config.yaml):

| Section      | Key Parameters                                         |
|--------------|--------------------------------------------------------|
| `paths`      | Data directories, checkpoint and output paths          |
| `data`       | Max sequence lengths, vocabulary min frequency         |
| `model`      | Embedding dims, hidden dims, dropout, attention toggle |
| `training`   | Batch size, learning rate, epochs, scheduler, patience |
| `llm`        | Gemini/Anthropic models and token budget for `/predict/analyze` |
| `serving`    | Host, port, model version, API key, cache TTL          |
| `database`   | MongoDB URI, database name                             |

Environment variable overrides for Docker: `MONGODB_URI`, `MONGODB_DB`, `GEMINI_API_KEY` or `ANTHROPIC_API_KEY` (enables AI analysis), `GEMINI_MODEL` / `ANTHROPIC_MODEL` (optional model overrides).

---

## Testing

```bash
# Run all unit tests (excluding MongoDB-dependent tests)
python -m pytest tests/ -v --ignore=tests/test_db.py

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Run a specific test file
python -m pytest tests/test_model.py -v

# Lint
ruff check src/ scripts/ tests/ --select E,F,W --ignore E501,E402
```

Test coverage includes:
- Text cleaning edge cases and category parsing
- Vocabulary building, encoding, serialization
- Model forward/backward pass and single-batch overfitting
- Loss function correctness and gradient flow
- API endpoint validation, error states, and input constraints
- MongoDB CRUD operations and aggregations (requires local MongoDB)

---

## License

MIT
