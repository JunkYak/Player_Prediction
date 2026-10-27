# 🏀 UN-REAL: NBA Player-Rating Prediction & Lineup Builder

An end-to-end, production-grade NBA player performance rating and prediction platform. UN-REAL transforms complex, high-velocity play-by-play telemetry into granular player-game performance ratings, predicts future rating momentum using an ensemble machine learning model, and delivers interactive draft and lineup optimization tools via a modern web interface.

---

## 1. Project Overview

**UN-REAL** is designed to evaluate, rank, and project NBA player performance using real play-by-play dynamics rather than simplistic box-score averages. The system addresses critical analytical needs across fantasy drafting, roster construction, and player momentum tracking:

- **Data-Driven Player Ratings**: Computes objective, game-by-game impact ratings evaluating shot difficulty, playmaking leverage, defensive disruption, context changes, and high-leverage 4th-quarter performance.
- **Form Momentum Predictions**: Forecasts next-game player ratings utilizing historical multi-game form windows (`last_game`, `last3_avg`, `last5_avg`, `last7_avg`, `expand_mean`) and schedule rest factors.
- **Interactive Lineup Builder**: Enables users to draft 5-slot starting lineups, evaluate combined projected rating outputs, and analyze team composition.
- **League-Wide Player & Team Explorer**: Full interactive access to all 30 NBA teams, active rosters, player depth charts, and detailed game logs.
- **Automated Production Serving**: Runs a serverless daily pipeline on Azure Functions to discover new games, normalize play-by-play data with automatic failover, update ratings, and publish atomic, validated serving snapshots to Azure Blob Storage.

---

## 2. Features

### Frontend Capabilities (React + Vite)
- **Interactive Draft Lineup Builder**: 5-slot starting roster picker allowing dynamic player assignment, quick replacement, individual slot clearing, and instantaneous composite rating calculation.
- **Team & Roster Explorer**: Search and browse all 30 NBA franchises, complete with official team branding, color schemes, and full active rosters.
- **Top Players & Slate Filtering**: View active draft slates for upcoming game dates, sort by projected rating, and filter by player availability.
- **Player Detail & Analytics Modal**: Deep-dive analytics modal showing historical rating trajectories, recent form averages, days of rest, back-to-back indicators, headshots, and injury statuses.
- **Live Injury Status Badges**: Real-time player availability indicators (`Active`, `Questionable`, `Doubtful`, `Out`) integrated directly into player cards and drafting interfaces.
- **Responsive Dark-Themed UI**: Built with custom high-performance CSS tokens, smooth transitions, glassmorphic accents, and intuitive mobile navigation.

### Backend Capabilities (FastAPI)
- **High-Performance Parquet Serving**: In-memory serving from columnar Apache Parquet snapshots with microsecond query responses.
- **Automated Snapshot Hot-Reloading**: Seamless background detection and reloading of updated serving snapshots via Azure Blob Storage manifests without process restarts or service downtime.
- **Contract Endpoints**:
  - `GET /health`: Liveness probe for orchestration health checks.
  - `GET /ready`: Readiness probe verifying serving artifacts are loaded in memory and healthy.
  - `GET /status`: Diagnostic endpoint reporting slate counts, total players, snapshot timestamps, and pipeline staleness.
  - `GET /api/v1/players`: Returns active draft slate players for the target game slate.
  - `GET /api/v1/teams`: Returns league-wide player roster and rating projections across all teams.
- **CORS & Multi-Environment Support**: Configurable origin controls for local development, staging, and production hosting.

---

## 3. Architecture

UN-REAL employs a decoupled, cloud-native architecture that strictly separates scheduled batch ingestion from the live serving layer.

```
Upstream NBA Telemetry
┌─────────────────────────┐         ┌─────────────────────────┐
│   NBA Stats API (Primary)│         │   ESPN API / CDN (Fallback)
└────────────┬────────────┘         └────────────┬────────────┘
             │                                   │
             └─────────────────┬─────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ Azure Functions (`propredict-daily-pipeline`)               │
│ - Daily Timer Trigger: 07:30 UTC / 13:00 IST                │
│ - Discovery & 14-Column PBP Normalization                   │
│ - Deduplication & Atomic Parquet Write                      │
│ - Rating Engine Formula & Rolling Form Features             │
│ - Random Forest ML Inference                                │
│ - Policy A Fail-Closed Pre-Publication Validation           │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼ (Canonical Data)             ▼ (Immutable Snapshots)
┌──────────────────────────────┐  ┌──────────────────────────────────────┐
│ Azure Blob Storage           │  │ Azure Blob Storage                   │
│ Container: propredict-history│  │ Container: propredict-serving        │
│ (`play_by_play.parquet`)     │  │ (`snapshots/<id>/`, `current_manifest`)│
└──────────────────────────────┘  └──────────────────┬───────────────────┘
                                                     │ Polling / Hot-Reload
                                                     ▼
                                  ┌──────────────────────────────────────┐
                                  │ FastAPI Backend (`propredict-nba`)   │
                                  │ - Azure App Service (Python 3.11)    │
                                  │ - In-Memory Parquet State            │
                                  │ - Endpoints: /ready, /api/v1/players │
                                  └──────────────────┬───────────────────┘
                                                     │ JSON REST
                                                     ▼
                                  ┌──────────────────────────────────────┐
                                  │ React/Vite Frontend (Vercel)         │
                                  │ - URL: https://un-real-ten.vercel.app│
                                  │ - Draft Lineup, Teams, Analytics     │
                                  └──────────────────────────────────────┘
```

### Component Roles & Interaction
1. **Data Ingestion Pipeline (`propredict-daily-pipeline`)**: Runs on Azure Functions (Linux, Python 3.11). Executes daily batch processing in an isolated `/tmp` workspace, downloading baseline data, ingesting completed games, calculating ratings, and validating artifacts.
2. **Canonical Data Storage (`propredict-history`)**: Houses the long-term historical baseline (`play_by_play.parquet`), incrementally updated as completed games are verified.
3. **Serving Layer Storage (`propredict-serving`)**: Stores immutable point-in-time snapshot directories (`snapshots/<snapshot_id>/`) containing `frontend_home.parquet`, `frontend_all.parquet`, and `pipeline_status.json`, managed by the single pointer `current_manifest.json`.
4. **Live REST API (`propredict-nba`)**: Hosted on Azure App Service. Periodically polls `current_manifest.json` for new snapshot releases, downloads updated Parquet files to an internal cache, and serves live HTTP queries.
5. **Client Application**: Single-page application hosted on Vercel connecting to the Azure App Service REST API over HTTPS.

---

## 4. Prediction Pipeline & Safeguards

The pipeline processes games through seven structured phases:

1. **Game Discovery**:
   - `scripts/get_next_day_games.py` scans target schedule slates to detect finalized games requiring ingestion and upcoming games requiring predictions.
2. **Telemetry Ingestion & Schema Normalization**:
   - `scripts/build_play_by_play_dataset.py` fetches box scores and play-by-play feeds.
   - **Primary Provider**: NBA Stats API (`nba_api`).
   - **Fallback Provider**: If NBA Stats encounters connection errors or HTTP timeouts, the pipeline automatically falls back to the ESPN CDN (`scripts/espn_pbp_adapter.py`).
   - **14-Column Normalization**: Incoming ESPN play feeds are strictly normalized to match the canonical 14-column schema (`game_id`, `event_num`, `period`, `clock`, `scoreHome`, `scoreAway`, etc.), casting scores cleanly to strings (`clean_score_str`) to prevent mixed-type Arrow serialization exceptions.
3. **Incremental Deduplication & Atomic Persistence**:
   - Deduplicates incoming events against historical records using composite key `(game_id, event_num)`.
   - Writes datasets atomically via temporary files (`.tmp`) and file replacement to ensure zero data corruption during unexpected interruptions.
4. **Player Rating Computation**:
   - `scripts/compute_player_ratings.py` processes raw event logs through the domain rating engine (`rating_engine/formula.py`), evaluating offensive creation, efficiency, defensive stops, assists, turnovers, and 4th-quarter clutch performance.
5. **Feature Engineering**:
   - `scripts/build_features.py` compiles multi-game rolling windows and player momentum trends strictly enforcing the **February 6, 2026** historical cutoff date.
6. **Inference**:
   - `scripts/predict_ratings.py` feeds the 7 strictly ordered features into the pre-trained Random Forest model (`models/rating_model.pkl`).
7. **Roster & Injury Reconciliation**:
   - `scripts/injury_provider.py` scrapes official NBA injury reports and reconciles player designations (`Active`, `Questionable`, `Doubtful`, `Out`) with upcoming game slates.

### Policy A: Fail-Closed Publication Safeguard
To guarantee zero production downtime or degradation, the serving publication workflow enforces **Policy A**:
- **Atomic Validation**: Before touching production, the pipeline validates artifact completeness: non-empty rows, required schema columns, valid snapshot timestamps, and correct historical cutoff dates.
- **Fail-Closed Isolation**: If any step fails (e.g., upstream NBA API outage, partial ingestion, validation error, schema mismatch), **publication is immediately halted**.
- **Preservation of Last-Good State**: `current_manifest.json` is never updated on failure. The production API and frontend continue serving the last-known-good serving snapshot uninterrupted.

---

## 5. Data and Model Specifications

### Model Architecture
- **Model Type**: Scikit-learn `RandomForestRegressor`
- **Location**: `models/rating_model.pkl` (SHA-256: `71e19ab50e35d6bf0dbdca92599b1b3f5bcca4dc17d3c614f69b1b0727a4bf11`)
- **Metadata**: `models/feature_metadata.json`
- **Inference Mode**: Static pre-trained inference. The model is intentionally **not** retrained automatically in daily scheduled runs to maintain deterministic and consistent scoring dynamics.

### Verified Feature Schema (Strict Order)
The inference engine and model artifact require exactly 7 features in the following precise sequence:

| Index | Feature Name | Description |
|:-----:|:-------------|:------------|
| 1 | `last_game` | Player rating achieved in the most recent game played |
| 2 | `last3_avg` | Rolling arithmetic mean rating over the last 3 games |
| 3 | `last5_avg` | Rolling arithmetic mean rating over the last 5 games |
| 4 | `last7_avg` | Rolling arithmetic mean rating over the last 7 games |
| 5 | `games_played_before_target` | Total qualifying games logged by the player prior to target game |
| 6 | `expand_mean` | Expanding lifetime mean rating across all observed historical games |
| 7 | `days_since_prev_game` | Number of calendar days elapsed since the player's prior game |

### Historical Data Cutoff
- **Cutoff Date**: `2026-02-06`
- **Enforcement**: Only games played on or after February 6, 2026 are included in training and feature calculation routines. All ingestion adapters, feature builders, and publication validators actively assert this boundary.

---

## 6. Technology Stack

### Backend & Machine Learning
- **Python**: 3.11 / 3.13
- **Framework**: FastAPI (0.111+), Starlette, Uvicorn, Gunicorn
- **Data & ML**: Pandas (2.0+), NumPy (1.24+), Scikit-learn (1.3+), PyArrow (12.0+), Joblib
- **Upstream Data**: `nba_api`, `nbainjuries`, Requests, Urllib3
- **Testing**: Pytest (8.0+), Unittest Mock

### Frontend
- **Framework**: React 18
- **Build Tool**: Vite
- **Icons**: Lucide React
- **Styling**: Vanilla CSS with customized design system tokens

### Cloud & DevOps
- **Azure Functions**: Consumption plan (Linux Python 3.11) for daily batch processing (`propredict-daily-pipeline`)
- **Azure App Service**: B1 Basic Linux App Service hosting FastAPI (`propredict-nba`)
- **Azure Blob Storage**: Blob containers `propredict-history` and `propredict-serving` (`azurefunceesh`)
- **Frontend Hosting**: Vercel (`un-real-ten.vercel.app`)

---

## 7. Repository Structure

```
Player-Prediction/
├── Readme.md                          # Comprehensive project documentation
├── requirements.txt                   # Production Python dependencies
├── run_full_pipeline.py               # Local orchestrator for end-to-end pipeline execution
├── orchestrator.py                    # Production daily pipeline runner with single-writer claims
├── Dockerfile                         # Container definition for API deployment
├── Dockerfile.job                     # Dedicated batch runner for Azure Container Apps / jobs
│
├── api/                               # FastAPI serving application
│   ├── config.py                      # App settings and environment variable bindings
│   ├── loader.py                      # Parquet loader, Azure Blob polling, and serving state
│   ├── main.py                        # FastAPI application factory, lifespan, CORS, static mounts
│   └── routers/
│       ├── health.py                  # Liveness (/health), readiness (/ready), diagnostics (/status)
│       └── players.py                 # Draft slate (/api/v1/players) and all players (/api/v1/teams)
│
├── azure_function/                    # Standalone Azure Function App deployment bundle
│   ├── function_app.py                # Timer trigger (07:30 UTC) and HTTP trigger (/trigger_refresh)
│   ├── host.json                      # Azure Functions runtime configuration
│   ├── requirements.txt               # Function App Python dependencies
│   ├── .funcignore                    # Excluded files during deployment
│   ├── api/                           # Mirrored API modules for function execution
│   ├── rating_engine/                 # Mirrored rating formula modules
│   ├── scripts/                       # Mirrored orchestration and data scripts
│   └── models/                        # Mirrored model artifact and feature metadata
│
├── data/                              # Canonical baseline datasets (Parquet ignored in Git)
│   ├── frontend_all.json              # Static fallback seed for league-wide players
│   ├── frontend_home.json             # Static fallback seed for slate players
│   └── team_rankings.json             # Team ranking metadata and depth charts
│
├── lineup-builder/                    # React / Vite frontend application
│   ├── index.html                     # Frontend entrypoint
│   ├── package.json                   # Node.js dependencies and scripts
│   ├── vite.config.js                 # Vite build and dev server configuration
│   └── src/
│       ├── App.jsx                    # Root component, state management, modal controllers
│       ├── App.css                    # Design system styling, components, responsive layout
│       ├── api.js                     # API client connecting to FastAPI backend
│       ├── components/
│       │   ├── AnalyticsSection.jsx   # League analytics and player performance trends
│       │   ├── LineupSection.jsx      # Interactive 5-slot lineup drafter
│       │   ├── NbaLogo.jsx            # SVG NBA team logo components
│       │   ├── PlayerCard.jsx         # Card component with rating, injury tag, and actions
│       │   ├── PlayerDetailModal.jsx  # Detailed player breakdown and historical game log
│       │   ├── Slot.jsx               # Individual roster slot component
│       │   ├── SlotPickerModal.jsx    # Slot selection dialog
│       │   ├── TeamsSection.jsx       # 30-team browser and roster depth charts
│       │   └── TopPlayersSection.jsx  # Top players leaderboard and position filter
│       └── constants/
│           └── teams.js               # Team metadata, color palettes, and abbreviations
│
├── models/                            # Machine learning model artifacts
│   ├── rating_model.pkl               # Canonical trained Random Forest model (versioned)
│   └── feature_metadata.json          # Feature schema and median baseline statistics
│
├── rating_engine/                     # Core domain rating math
│   └── formula.py                     # Deterministic player rating calculation engine
│
├── scripts/                           # Core data pipeline and utility modules
│   ├── add_game_dates.py              # Attaches validated game dates and asserts cutoff
│   ├── apply_injury_status.py         # Merges injury tags with prediction slates
│   ├── azure_blob.py                  # Azure Blob storage synchronization and Policy A publication
│   ├── build_features.py              # Computes 7-feature rolling form representations
│   ├── build_frontend_dataset.py      # Aggregates frontend parquet and json artifacts
│   ├── build_play_by_play_dataset.py  # Ingestion engine, ESPN fallback, and atomic write
│   ├── check_schedule.py              # Schedule window guard for pipeline automation
│   ├── compute_player_ratings.py      # Computes game ratings from raw play events
│   ├── espn_pbp_adapter.py            # ESPN CDN fetcher and 14-column schema normalizer
│   ├── get_next_day_games.py          # Discovers completed and upcoming schedule slates
│   ├── injury_provider.py             # Official NBA injury report scraper and classifier
│   ├── orchestrator.py                # Pipeline execution controller and single-writer locks
│   ├── predict_ratings.py             # Model inference engine enforcing feature ordering
│   └── roster_provider.py            # Active NBA roster scraping and player mapping
│
└── tests/                             # Comprehensive test suite (347 automated tests)
    ├── test_aca_job.py                # Tests for containerized batch job execution
    ├── test_blob_sync.py              # Tests for Azure Blob synchronization and manifests
    ├── test_freshness_failure_simulation.py # Staleness and degradation handling tests
    ├── test_incremental_ingestion_and_cutoff.py # Tests for duplicate handling and cutoff
    ├── test_pbp_schema_normalization_and_merge.py # Normalization and Arrow type tests
    ├── test_policy_a_fail_closed_publication.py # Fail-closed publication security tests
    ├── test_production_release.py     # Production FastAPI and server contract tests
    ├── test_rating_engine.py          # Regression tests for rating formula accuracy
    ├── test_roster_updates.py         # Roster provider and injury tagging tests
    └── test_slate_selection.py        # Slate discovery and game scheduling tests
```

---

## 8. Local Setup & Quickstart

### Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- Node.js 18+ and npm
- Git

### 1. Clone Repository
```bash
git clone https://github.com/JunkYak/Player_Prediction.git
cd Player_Prediction
```

### 2. Backend Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Local Environment Variables
Create a `.env` file in the project root:
```ini
PROPREDICT_ENV=development
PROPREDICT_DATA_DIR=./data
PROPREDICT_ALLOWED_ORIGINS=http://localhost:5173
PROPREDICT_RELOAD=false

# Optional: To connect to Azure Blob Storage locally (requires az login)
AZURE_STORAGE_ACCOUNT=azurefunceesh
AZURE_SERVING_CONTAINER=propredict-serving
AZURE_MANIFEST_NAME=current_manifest.json
AZURE_BLOB_CHECK_INTERVAL_SECONDS=60
```

### 4. Running the Backend API
```bash
uvicorn api.main:app --reload --port 8000
```
- Access API documentation: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Readiness check: `http://localhost:8000/ready`

### 5. Running the Pipeline Locally
To run a complete end-to-end local data refresh:
```bash
python run_full_pipeline.py
```
To run predictions without fetching new external data:
```bash
python run_full_pipeline.py --skip-fetch
```

### 6. Frontend Setup
In a new terminal:
```bash
cd lineup-builder
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 9. Azure Deployment

The production environment is hosted on Microsoft Azure:

- **Azure Function App (`propredict-daily-pipeline`)**:
  - Hosts the scheduled pipeline on Linux Consumption (Python 3.11).
  - Configured with Timer Trigger `0 30 7 * * *` (07:30 UTC / 13:00 IST daily).
  - Provides authenticated HTTP endpoint `/api/trigger_refresh` for on-demand syncs.
- **Azure App Service (`propredict-nba`)**:
  - Hosts the FastAPI serving application on Linux B1 Basic.
  - Automatically loads and polls snapshot artifacts from Azure Blob Storage.
- **Azure Blob Storage (`azurefunceesh`)**:
  - `propredict-history`: Retains canonical historical Parquet telemetry.
  - `propredict-serving`: Holds immutable serving snapshots and atomic `current_manifest.json`.
- **Vercel Frontend**:
  - Deployed at `https://un-real-ten.vercel.app`, querying the live App Service API.

### Azure Security & Access Pattern
- Production services communicate securely via Azure Managed Identities and Entra ID (OIDC) roles (`Storage Blob Data Contributor` and `Storage Blob Data Reader`).
- No storage keys, connection strings, or function master keys are stored in code or repository configuration.

---

## 10. Testing

The repository maintains a rigorous automated testing suite covering unit math, schema invariants, fail-closed publication, and integration contracts.

### Executing Tests
Run the complete test suite:
```bash
python -m pytest
```
Run with concise output:
```bash
python -m pytest -q
```
Run specific test modules:
```bash
python -m pytest tests/test_pbp_schema_normalization_and_merge.py
python -m pytest tests/test_policy_a_fail_closed_publication.py
python -m pytest tests/test_production_release.py
```

### Verified Test Results
- **Status**: 347 passed, 0 failures, 3 warnings.
- **Key Test Coverage Areas**:
  - Deterministic rating engine calculations and regression baselines.
  - ESPN play-by-play schema normalization, `clean_score_str` type safety, and Arrow serialization.
  - Incremental ingestion deduplication and February 6, 2026 cutoff boundary enforcement.
  - Policy A fail-closed publication prevention under partial, corrupted, or missing inputs.
  - FastAPI serving routes, readiness state management, and CORS middleware behavior.

---

## 11. Current Limitations

- **Upstream Telemetry Timeouts**: The official NBA Stats API periodically encounters rate limits, connection drops, or latency spikes. While the ESPN fallback mitigates outages, ESPN play-by-play data contains minor descriptive differences that require continuous normalization.
- **Momentum vs. Full Tactical Context**: The rating model focuses on recent form, momentum, and schedule rest. It does not explicitly model in-game opponent defensive matchups, defensive player tracking, or game-by-game coaching tactical changes.
- **Player Rating Scope**: Ratings measure individualized player impact and momentum; they are not a direct point-spread or team win-loss prediction model.

---

## 12. Future Improvements

- **Matchup Difficulty Modeling**: Integrate opposing team defensive ratings by position to dynamically adjust projected player efficiency.
- **Minutes & Usage Rate Projection**: Incorporate projected playing time and team injury vacated usage into raw rating scaling.
- **Ensemble Upgrades**: Benchmark Gradient Boosting (LightGBM / XGBoost) implementations against the existing Random Forest baseline while preserving historical feature boundaries.
- **Live In-Game Tracking**: Real-time score and play-by-play websocket streams during live game windows.