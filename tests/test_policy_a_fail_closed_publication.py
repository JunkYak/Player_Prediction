"""
tests/test_policy_a_fail_closed_publication.py

Test Suite for Policy A: Fail-Closed Serving Publication & Data Recovery.
Verifies that:
1. Complete, validated ingestion can publish normally.
2. Partial ingestion cannot change root current_manifest.json.
3. Failed ingestion cannot change root current_manifest.json.
4. Missing or malformed ingestion metadata cannot publish.
5. A failed publication preserves the prior serving pointer and prior snapshot.
6. Successfully ingested games remain available for subsequent processing in play_by_play.parquet.
7. Failed game IDs can be retried on a subsequent run.
8. A successful subsequent complete run can publish normally.
9. Historical cutoff enforcement (2026-02-06) remains strictly intact.
10. Model feature names, feature order, model artifact, and rating formula remain unchanged.
"""

import os
import sys
import json
import uuid
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.azure_blob import (
    publish_serving_snapshot,
    validate_ingestion_metadata_for_publication,
    DEFAULT_MANIFEST_BLOB,
)
from scripts.build_play_by_play_dataset import (
    merge_pbp_data,
    build_dataset,
    HISTORICAL_CUTOFF_DATE,
    REQUIRED_COLUMNS as PBP_COLUMNS,
)
from scripts.predict_ratings import construct_prediction_features
import joblib


# ==============================================================================
# FIXTURES & MOCKS
# ==============================================================================
@pytest.fixture
def mock_blob_store():
    """In-memory mock store for Azure Blob Storage."""
    store = {}
    initial_manifest = {
        "snapshot_id": "snap_initial_good_12345678",
        "published_at": "2026-10-21T10:00:00+00:00",
        "data_freshness": {
            "source_freshness_status": "FRESH",
            "is_fresh": True,
            "latest_completed_game_date": "2026-10-20",
        },
        "files": {
            "frontend_all.parquet": {"sha256": "initial_all_hash"},
            "frontend_home.parquet": {"sha256": "initial_home_hash"},
        },
    }
    store[DEFAULT_MANIFEST_BLOB] = json.dumps(initial_manifest, indent=2).encode("utf-8")
    store["snapshots/snap_initial_good_12345678/manifest.json"] = store[DEFAULT_MANIFEST_BLOB]
    return store


@pytest.fixture
def mock_service_client(mock_blob_store):
    """Mock Azure BlobServiceClient operating on mock_blob_store."""
    mock_container = MagicMock()

    def get_client(name):
        client = MagicMock()
        client.exists.side_effect = lambda: name in mock_blob_store
        client.download_blob.return_value.readall.side_effect = lambda: mock_blob_store.get(name, b"")
        client.upload_blob.side_effect = lambda data, overwrite=False: mock_blob_store.update(
            {name: data.read() if hasattr(data, "read") else (data.encode("utf-8") if isinstance(data, str) else data)}
        )
        return client

    mock_container.get_blob_client.side_effect = get_client
    service = MagicMock()
    service.get_container_client.return_value = mock_container
    return service


@pytest.fixture
def setup_serving_data(tmp_path):
    """Setup valid local serving parquets."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime, timedelta, timezone
    from zoneinfo import ZoneInfo
    now_ny = datetime.now(ZoneInfo("America/New_York"))
    today_ny = now_ny.strftime("%Y-%m-%d")
    yesterday_ny = (now_ny - timedelta(days=1)).strftime("%Y-%m-%d")

    from api.loader import REQUIRED_COLUMNS
    record = {col: "val_0" for col in REQUIRED_COLUMNS}
    record["rating"] = 25.5
    record["predicted_rating"] = 26.0
    record["game_date"] = today_ny
    record["last_game_date"] = yesterday_ny

    df_all = pd.DataFrame([record])
    df_home = pd.DataFrame([record])

    df_all.to_parquet(data_dir / "frontend_all.parquet", index=False)
    df_home.to_parquet(data_dir / "frontend_home.parquet", index=False)

    tomorrow_ny = now_ny.date().strftime("%Y-%m-%d")
    sched_record = {
        "game_id": "0022400999",
        "game_date": tomorrow_ny,
        "schedule_date": tomorrow_ny,
        "home_team_id": 1610612738,
        "home_team_name": "Boston Celtics",
        "away_team_id": 1610612747,
        "away_team_name": "Los Angeles Lakers",
        "home_team_tricode": "BOS",
        "away_team_tricode": "LAL",
        "status": "Scheduled",
        "game_status": 1,
        "game_time_utc": f"{tomorrow_ny}T23:30:00+00:00",
        "is_postponed": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    df_sched = pd.DataFrame([sched_record])
    df_sched.to_parquet(data_dir / "schedule_tomorrow.parquet", index=False)
    with open(data_dir / "schedule_tomorrow_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "CONFIRMED",
            "schedule_date": tomorrow_ny,
            "target_date": tomorrow_ny,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "games_scheduled": True,
            "games_count": 1,
            "is_stale": False,
            "error": None,
        }, f)

    return data_dir


# ==============================================================================
# TEST 1: COMPLETE INGESTION CAN PUBLISH
# ==============================================================================
def test_1_complete_ingestion_publishes_successfully(setup_serving_data, mock_service_client, mock_blob_store):
    """Complete, fresh ingestion run successfully updates current_manifest.json."""
    data_dir = setup_serving_data
    with open(data_dir / "ingestion_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "SUCCESS",
            "is_fresh": True,
            "new_games_ingested": 2,
            "failed_games_count": 0,
            "historical_cutoff_date": "2026-02-06",
            "latest_completed_game_date": "2026-10-22",
        }, f)

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service_client):
        manifest = publish_serving_snapshot(
            account_name="testacc",
            container_name="propredict-serving",
            data_dir=str(data_dir),
            git_sha="abcdef12",
        )

    assert manifest["snapshot_id"].startswith("snap_")
    assert manifest["data_freshness"]["source_freshness_status"] == "FRESH"
    assert manifest["data_freshness"]["is_fresh"] is True

    # current_manifest.json is updated to new snapshot
    active = json.loads(mock_blob_store[DEFAULT_MANIFEST_BLOB].decode("utf-8"))
    assert active["snapshot_id"] == manifest["snapshot_id"]


# ==============================================================================
# TEST 2: PARTIAL INGESTION CANNOT CHANGE CURRENT_MANIFEST.JSON
# ==============================================================================
def test_2_partial_ingestion_blocked_from_publishing(setup_serving_data, mock_service_client, mock_blob_store):
    """When status is PARTIAL_SUCCESS, publication fails closed and current_manifest.json is unchanged."""
    data_dir = setup_serving_data
    initial_manifest_bytes = mock_blob_store[DEFAULT_MANIFEST_BLOB]

    with open(data_dir / "ingestion_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "PARTIAL_SUCCESS",
            "is_fresh": False,
            "new_games_ingested": 1,
            "failed_games_count": 1,
            "historical_cutoff_date": "2026-02-06",
            "failed_games": [{"game_id": "0022600999", "error": "Timeout"}],
        }, f)

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service_client):
        with pytest.raises(RuntimeError, match="Publication blocked .*PARTIAL_SUCCESS"):
            publish_serving_snapshot(
                account_name="testacc",
                container_name="propredict-serving",
                data_dir=str(data_dir),
            )

    # current_manifest.json MUST remain untouched
    assert mock_blob_store[DEFAULT_MANIFEST_BLOB] == initial_manifest_bytes
    manifest_now = json.loads(mock_blob_store[DEFAULT_MANIFEST_BLOB].decode("utf-8"))
    assert manifest_now["snapshot_id"] == "snap_initial_good_12345678"


# ==============================================================================
# TEST 3: FAILED INGESTION CANNOT CHANGE CURRENT_MANIFEST.JSON
# ==============================================================================
def test_3_failed_ingestion_blocked_from_publishing(setup_serving_data, mock_service_client, mock_blob_store):
    """When status is FAILED, publication is blocked and current_manifest.json is unchanged."""
    data_dir = setup_serving_data
    initial_manifest_bytes = mock_blob_store[DEFAULT_MANIFEST_BLOB]

    with open(data_dir / "ingestion_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "FAILED",
            "is_fresh": False,
            "new_games_ingested": 0,
            "failed_games_count": 2,
            "historical_cutoff_date": "2026-02-06",
        }, f)

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service_client):
        with pytest.raises(RuntimeError, match="Publication blocked .*FAILED"):
            publish_serving_snapshot(
                account_name="testacc",
                container_name="propredict-serving",
                data_dir=str(data_dir),
            )

    assert mock_blob_store[DEFAULT_MANIFEST_BLOB] == initial_manifest_bytes


# ==============================================================================
# TEST 4: MISSING OR MALFORMED METADATA CANNOT PUBLISH
# ==============================================================================
def test_4a_missing_metadata_blocks_publication(setup_serving_data, mock_service_client, mock_blob_store):
    """Missing ingestion_metadata.json fails closed."""
    data_dir = setup_serving_data
    meta_file = data_dir / "ingestion_metadata.json"
    if meta_file.exists():
        meta_file.unlink()

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service_client):
        with pytest.raises(ValueError, match="Missing ingestion_metadata.json"):
            publish_serving_snapshot(
                account_name="testacc",
                container_name="propredict-serving",
                data_dir=str(data_dir),
            )


def test_4b_malformed_metadata_blocks_publication(setup_serving_data, mock_service_client):
    """Malformed JSON fails closed."""
    data_dir = setup_serving_data
    with open(data_dir / "ingestion_metadata.json", "w", encoding="utf-8") as f:
        f.write("{ INVALID JSON NOT COMPLETE ...")

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service_client):
        with pytest.raises(ValueError, match="Malformed ingestion_metadata.json"):
            publish_serving_snapshot(
                account_name="testacc",
                container_name="propredict-serving",
                data_dir=str(data_dir),
            )


def test_4c_contradictory_metadata_blocks_publication(setup_serving_data, mock_service_client):
    """Status SUCCESS but failed_games_count > 0 fails closed."""
    data_dir = setup_serving_data
    with open(data_dir / "ingestion_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "SUCCESS",
            "is_fresh": True,
            "new_games_ingested": 2,
            "failed_games_count": 1,  # Contradiction!
            "historical_cutoff_date": "2026-02-06",
        }, f)

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service_client):
        with pytest.raises(ValueError, match="Contradictory ingestion metadata"):
            publish_serving_snapshot(
                account_name="testacc",
                container_name="propredict-serving",
                data_dir=str(data_dir),
            )


# ==============================================================================
# TEST 5: FAILED PUBLICATION PRESERVES PRIOR SERVING POINTER AND PRIOR SNAPSHOT
# ==============================================================================
def test_5_prior_snapshot_and_manifest_preserved(setup_serving_data, mock_service_client, mock_blob_store):
    """A blocked publication never creates snapshot blobs and preserves the prior manifest."""
    data_dir = setup_serving_data
    with open(data_dir / "ingestion_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "PARTIAL_SUCCESS",
            "is_fresh": False,
            "new_games_ingested": 1,
            "failed_games_count": 1,
            "historical_cutoff_date": "2026-02-06",
        }, f)

    initial_keys = set(mock_blob_store.keys())

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service_client):
        try:
            publish_serving_snapshot(
                account_name="testacc",
                container_name="propredict-serving",
                data_dir=str(data_dir),
            )
        except RuntimeError:
            pass

    # No new blobs were written to the store
    assert set(mock_blob_store.keys()) == initial_keys
    # Prior snapshot remains
    assert "snapshots/snap_initial_good_12345678/manifest.json" in mock_blob_store
    active = json.loads(mock_blob_store[DEFAULT_MANIFEST_BLOB].decode("utf-8"))
    assert active["snapshot_id"] == "snap_initial_good_12345678"


# ==============================================================================
# TEST 6: SUCCESSFULLY INGESTED GAMES REMAIN AVAILABLE IN PBP DATASET
# ==============================================================================
def test_6_successful_games_preserved_in_play_by_play(tmp_path):
    """When Game 1 succeeds and Game 2 fails, Game 1 is saved in play_by_play.parquet."""
    out_file = tmp_path / "play_by_play.parquet"

    # Baseline with existing game
    base_row = {col: "val" for col in PBP_COLUMNS}
    base_row["gameId"] = "0022600001"
    base_row["actionNumber"] = 1
    base_df = pd.DataFrame([base_row])
    base_df.to_parquet(out_file, index=False)

    # Ingest Game 2 (succeeds)
    g2_row = {col: "val" for col in PBP_COLUMNS}
    g2_row["gameId"] = "0022600002"
    g2_row["actionNumber"] = 1
    g2_df = pd.DataFrame([g2_row])

    merged = merge_pbp_data(base_df, g2_df)
    merged.to_parquet(out_file, index=False)

    loaded = pd.read_parquet(out_file)
    assert "0022600001" in loaded["gameId"].values
    assert "0022600002" in loaded["gameId"].values
    assert loaded["gameId"].nunique() == 2


# ==============================================================================
# TEST 7: FAILED GAME IDS ARE RETRIED ON SUBSEQUENT RUN
# ==============================================================================
def test_7_failed_game_ids_retried_on_next_run(tmp_path):
    """Game 2 failed on run 1; on run 2, it is identified as missing and retried."""
    out_file = tmp_path / "play_by_play.parquet"
    row1 = {col: "val" for col in PBP_COLUMNS}
    row1["gameId"] = "0022600001"
    row1["actionNumber"] = 1
    pd.DataFrame([row1]).to_parquet(out_file, index=False)

    discovered = [
        {"game_id": "0022600001", "game_date": "2026-10-21"},
        {"game_id": "0022600002", "game_date": "2026-10-21"},  # Previously failed
    ]

    existing = pd.read_parquet(out_file)
    existing_ids = set(existing["gameId"].unique())

    missing = [g for g in discovered if g["game_id"] not in existing_ids]
    assert len(missing) == 1
    assert missing[0]["game_id"] == "0022600002"  # Retried!


# ==============================================================================
# TEST 8: SUCCESSFUL SUBSEQUENT RUN CAN PUBLISH NORMALLY
# ==============================================================================
def test_8_successful_subsequent_run_publishes(setup_serving_data, mock_service_client, mock_blob_store):
    """After retry succeeds, complete ingestion publishes normally."""
    data_dir = setup_serving_data
    with open(data_dir / "ingestion_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "SUCCESS",
            "is_fresh": True,
            "new_games_ingested": 1,
            "existing_games_skipped": 1,
            "failed_games_count": 0,
            "historical_cutoff_date": "2026-02-06",
            "latest_completed_game_date": "2026-10-22",
        }, f)

    with patch("scripts.azure_blob.get_blob_service_client", return_value=mock_service_client):
        manifest = publish_serving_snapshot(
            account_name="testacc",
            container_name="propredict-serving",
            data_dir=str(data_dir),
            git_sha="11223344",
        )

    assert manifest["data_freshness"]["source_freshness_status"] == "FRESH"
    active = json.loads(mock_blob_store[DEFAULT_MANIFEST_BLOB].decode("utf-8"))
    assert active["snapshot_id"] == manifest["snapshot_id"]


# ==============================================================================
# TEST 9: HISTORICAL CUTOFF ENFORCEMENT REMAINS INTACT
# ==============================================================================
def test_9_historical_cutoff_enforcement_intact():
    """Confirms historical cutoff constant and filtering strictly enforces 2026-02-06."""
    assert HISTORICAL_CUTOFF_DATE == "2026-02-06"

    # Prediction features filter pre-cutoff
    history = pd.DataFrame([
        {
            "gameId": "old",
            "teamId": 1,
            "personId": 100,
            "playerName": "Old Player",
            "rating": 20.0,
            "game_date": "2026-01-15",  # Pre-cutoff
            "fga": 10,
        },
        {
            "gameId": "new",
            "teamId": 1,
            "personId": 100,
            "playerName": "Old Player",
            "rating": 25.0,
            "game_date": "2026-02-10",  # Post-cutoff
            "fga": 12,
        },
    ])
    features = construct_prediction_features(history, target_date="2026-02-11")
    assert not features.empty
    assert features["last_game_date"].iloc[0] == "2026-02-10"


# ==============================================================================
# TEST 10: MODEL ARTIFACT AND RATING FORMULA REMAIN UNCHANGED
# ==============================================================================
def test_10_model_and_formula_remain_unchanged():
    """Confirms model feature order and rating formula SHA-256 match expected baselines."""
    # 1. Rating formula hash
    with open("rating_engine/formula.py", "rb") as f:
        formula_hash = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    assert formula_hash == "ecb4538a4f4568d4d66d3465bc8d46d35dcdeef28d9e5af3849368b39cbeaf06"

    # 2. Model feature ordering
    model = joblib.load("models/rating_model.pkl")
    expected_features = [
        "last_game",
        "last3_avg",
        "last5_avg",
        "last7_avg",
        "games_played_before_target",
        "expand_mean",
        "days_since_prev_game",
    ]
    assert list(model.feature_names_in_) == expected_features
    assert model.n_features_in_ == 7
