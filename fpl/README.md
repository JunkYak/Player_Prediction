# `fpl/` — Fantasy Premier League expected points

A self-contained package next to the NBA pipeline. It doesn't import NBA code,
and NBA code doesn't import it. Full design and PR sequence:
[implementation plan](https://claude.ai/artifact/L3BmoGiUaxv7AixfzFXN1b).

This is **PR 1: data + baselines**. The decomposed EP model (minutes, team
strength, attack, defensive contribution, saves, bonus, cards), the MILP
optimizer, the API router and the UI page come in later PRs.

## Setup

```bash
pip install -r requirements.txt -r requirements-fpl.txt
```

Everything downloaded or generated goes under `data/fpl/` (git-ignored;
override with `FPL_DATA_DIR`).

## Commands

| Command | What it does |
|---|---|
| `python -m fpl.backtest --model baselines` | Walk-forward baseline table for 2024-25 and 2025-26 (downloads the archive on first run) |
| `python -m fpl.backtest --model baselines --seasons 2026-27` | Same for the live season, rebuilt from the FPL API (~11 min cold, cached after) |
| `python -m fpl.ingest.snapshot` | Save a pre-deadline `bootstrap-static` (captures FPL's `ep_next`); run at least once per GW |
| `python -m fpl.ingest.snapshot --list` | Show captured gameweeks |
| `python -m fpl.ingest.archive 2024-25 2025-26` | Pre-download archive seasons |
| `python -m pytest -q tests/fpl` | FPL tests (offline; the scoring check against real data skips if the archive isn't downloaded) |

## Layout

```
fpl/
  config.py           paths, seasons, positions, XI limits
  scoring.yaml        FPL scoring rules (checked: 0 mismatches vs archive total_points)
  scoring.py          load rules, score stat lines
  ingest/archive.py   vaastav/Fantasy-Premier-League per-fixture data, normalised schema
  ingest/fpl_api.py   cached, ~1 req/s client for the public FPL API
  ingest/snapshot.py  pre-deadline bootstrap snapshots -> ep_next history
  features/build.py   leak-free player-GW feature table
  metrics.py          MAE/RMSE, per-GW Spearman, top-20 precision, calibration, Brier, XI points
  backtest.py         walk-forward backtest + baselines
notebooks/fpl_eda.ipynb
tests/fpl/
```

## Data notes

* **Leakage.** The archive's `xP` column is dropped on load: it is written
  after the gameweek. Every feature for GW *g* comes only from gameweeks
  before *g*, including team form inside a double gameweek
  (`tests/fpl/test_fpl_features.py::test_features_only_use_earlier_gameweeks`).
* **xG gaps.** FPL began publishing xG at 2022-23 GW16, but the archive
  stores GW1-15 as 0.0. Those are set to NaN on load, and per-90 rates only
  count minutes where the stat exists.
* **Defensive contribution** exists from 2025-26 (NaN before, and scored as
  0 points by the season overrides in `scoring.yaml`).
* **2026-27.** The archive stops at GW6, so the live season comes from
  `element-summary/{id}/`. Positions are the current ones (11 players were
  reclassified this season).
* **`ep_next`** has no historical source, so that baseline only covers
  gameweeks snapshotted with `fpl.ingest.snapshot`. Where and how often that
  job runs is still to be agreed.
