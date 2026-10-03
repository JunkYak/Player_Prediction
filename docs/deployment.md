# NBA Player Performance Prediction System — Deployment & Orchestration Guide

## 1. System Architecture Overview

The system is partitioned into three strictly decoupled layers:

```
[NBA API / External Data]
           │
           ▼
[Batch Pipeline / Orchestrator] ──(Atomic .tmp -> os.replace)──▶ [data/*.parquet]
(scripts/orchestrator.py)                                               ▲
(Isolated Daily Process)                                                │
                                                               (Read-only + mtime check)
                                                                        │
[React Frontend SPA] ◀──(REST API JSON)── [FastAPI Serving Layer] ──────┘
(Vite / Static Web Server)                (uvicorn api.main:app)
```

1. **Daily Batch Pipeline & Orchestrator (`scripts/orchestrator.py` / `run_full_pipeline.py`)**:
   - Executes incrementally after NBA games complete.
   - Discovers new games, fetches play-by-play, computes ratings, builds T+1 lag features, predicts next-game ratings, applies injury statuses, identifies the next slate, and atomically publishes serving artifacts.
   - Runs as an independent process (cron, Windows Task Scheduler, Kubernetes CronJob, or background daemon).
   - **Never embedded inside FastAPI**.

2. **Serving Layer (`api/`)**:
   - FastAPI application served via `uvicorn`.
   - Read-only layer over `data/frontend_all.parquet` and `data/frontend_home.parquet`.
   - Performs no ML inference, calls no external APIs, and does not schedule jobs.
   - Monitors artifact modification timestamps (`mtime`) on incoming requests to seamlessly reload fresh data in-memory without requiring process restarts.

3. **Frontend Application (`lineup-builder/`)**:
   - Production React SPA built with Vite.
   - Consumes dynamic REST endpoints (`/api/v1/players`, `/api/v1/teams`).
   - Configurable API base URL via `VITE_API_BASE_URL`.

---

## 2. Daily Product Loop

Every day, the system completes the canonical product cycle:

```
1. NBA games complete
       ↓
2. New game data becomes available
       ↓
3. Incremental ingestion (scripts/build_play_by_play_dataset.py)
       - Lookback discovery
       - Deduplication & merge with data/play_by_play.parquet
       ↓
4. Historical data & ratings updated (scripts/compute_player_ratings.py)
       - Possession rating deltas (base, context, clutch)
       ↓
5. T+1 features regenerated (scripts/add_game_dates.py, scripts/build_features.py)
       - Strict leakage-free lag features (last_game, last3_avg, last5_avg, last7_avg, etc.)
       ↓
6. Prediction model generates next-game predictions (scripts/predict_ratings.py)
       - Evaluates latest features with rating_model.pkl
       ↓
7. Injury / availability information applied (scripts/apply_injury_status.py)
       - Official NBA injury report mapping (Active, Probable, Questionable, Doubtful, Out)
       - Safe failure handling: failure raises InjuryProviderFailure and preserves existing data
       ↓
8. Next NBA slate identified (scripts/get_next_day_games.py)
       - Discovers games scheduled for tomorrow
       - If zero games scheduled (offseason), slate is cleanly empty (0 rows)
       ↓
9. Frontend serving artifacts regenerated atomically (scripts/build_frontend_dataset.py)
       - Writes frontend_all.parquet.tmp and frontend_home.parquet.tmp
       - Validates row counts and schema integrity
       - Atomically commits via os.replace
       ↓
10. FastAPI detects fresh artifacts (api/loader.py)
       - Checks parquet mtime; reloads ServingState in memory with zero downtime
       ↓
11. React displays fresh rankings (src/api.js)
       - User drafts 5 players with multipliers (2.0x, 1.8x, 1.6x, 1.4x, 1.2x)
```

---

## 3. Environment Variables & Configuration

### Backend / Orchestrator (`.env`)

| Variable | Description | Default |
|---|---|---|
| `PROPREDICT_DATA_DIR` | Absolute or relative path to parquet artifacts directory | `./data` |
| `PROPREDICT_ALLOWED_ORIGINS` | Comma-separated list of allowed CORS origins | `http://localhost:5173` |
| `PROPREDICT_ENV` | Environment mode (`production`, `development`) | `production` |
| `PROPREDICT_RELOAD` | Enable uvicorn reload (development only) | `false` |
| `PROPREDICT_SCHEDULE_TIME` | Daily scheduled run time in `HH:MM` format for orchestrator daemon | `06:00` |
| `PROPREDICT_EXPORT_STATIC_JSON` | Set to `true` to export legacy static JSON files to `lineup-builder/public/` | `false` |

### Frontend (`lineup-builder/.env`)

| Variable | Description | Default |
|---|---|---|
| `VITE_API_BASE_URL` | Base URL of the FastAPI backend without trailing slash | `http://localhost:8000` |

---

## 4. Execution & Operation Commands

### 4.1 FastAPI Serving Layer
```bash
# Production startup with uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

### 4.2 React Frontend Build
```bash
cd lineup-builder
npm run build
# Output bundle is written to lineup-builder/dist/
```
The resulting `dist/` directory can be served via Nginx, Caddy, AWS S3/CloudFront, Cloudflare Pages, or any static file host.

### 4.3 Daily Pipeline Orchestrator

#### A. One-Shot Execution (Cron, Windows Task Scheduler, GitHub Actions, AWS EventBridge)
```bash
# Canonical daily update (ingests new games, updates ratings & features, predicts, updates artifacts)
python orchestrator.py --once
```

#### B. Daemon Mode (Standalone background process / Container)
```bash
# Runs continuously, firing daily at 06:00 local time
python orchestrator.py --daemon --time 06:00
```

#### C. Optional Flags
- `--skip-fetch`: Skip NBA API ingestion (useful for testing or offseason dry-runs).
- `--force-retrain`: Re-train model weights (default skips training during daily runs to preserve inference latency).
- `--export-json`: Also output legacy static JSON to `lineup-builder/public/`.

---

## 5. Critical Data Safety & Failure Isolation

1. **Process Lock (`data/pipeline.lock`)**:
   - Prevents concurrent pipeline executions from overlapping or clobbering temporary files.
   - Detects and clears stale lockfiles if the previous process crashed.

2. **Atomic Artifact Publication**:
   - `build_frontend_dataset.py` writes output to `frontend_all.parquet.tmp` and `frontend_home.parquet.tmp`.
   - Both files are read back and validated for row count and schema integrity.
   - Files are committed via `os.replace`, guaranteeing that readers (FastAPI) never see an empty, half-written, or corrupt file.
   - If an error occurs, `.tmp` files are immediately deleted and existing production artifacts remain untouched.

3. **External Provider Failure Resilience**:
   - If NBA API discovery, play-by-play fetch, or the injury provider fails (e.g. HTTP 403 / 503 / timeout), the pipeline raises an exception and aborts.
   - The failure is logged to `data/pipeline_status.json`.
   - FastAPI continues serving the previous valid dataset without disruption.

---

## 6. Offseason / No-Games Handling

When no games are scheduled for the target date:
- `data/next_day_games.parquet` contains 0 rows.
- `data/frontend_home.parquet` contains 0 rows.
- `GET /api/v1/players` returns:
  ```json
  {
    "game_date": null,
    "generated_at": "2026-10-02T13:11:00",
    "games_scheduled": false,
    "players": []
  }
  ```
- `GET /api/v1/teams` remains fully available and populated with all 30 teams and 500+ players.
- The React frontend displays the informative offseason banner: *"No NBA games are scheduled for today. Check back tomorrow!"* while keeping roster browsing available.
