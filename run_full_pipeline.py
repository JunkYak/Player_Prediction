"""
run_full_pipeline.py

Canonical Production Pipeline Runner for NBA Player Performance Prediction System.

Modes:
  --mode full     : Runs complete pipeline (PBP Ingestion -> Ratings -> Dates -> Features -> Training -> Predictions -> Availability -> Schedule -> Frontend JSON -> Team Rankings)
  --mode train    : Runs training pipeline only (Ratings -> Dates -> Features -> Training)
  --mode predict  : Runs prediction pipeline only (Predictions -> Availability -> Schedule -> Frontend JSON -> Team Rankings)

Options:
  --skip-fetch    : In full mode, skip historical play-by-play API fetch
  --skip-train    : In full mode, skip model re-training
"""
import time
import os
import sys
import argparse
import importlib.util

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.build_play_by_play_dataset import build_dataset
from scripts.compute_player_ratings import compute_player_ratings
from scripts.add_game_dates import add_game_dates
from scripts.build_features import build_features
from scripts.train_model import train_model
from scripts.predict_ratings import predict_ratings
from scripts.apply_injury_status import apply_injury_status
from scripts.get_next_day_games import get_next_day_games
from scripts.build_frontend_dataset import build_frontend_dataset
from scripts.team_rankings import build_team_rankings


# ==============================
# RUN STEP HELPER
# ==============================

def run_step(name: str, func):
    print(f"\n🔹 {name}...")
    start = time.time()
    try:
        result = func()
        elapsed = round(time.time() - start, 2)
        print(f"✅ {name} completed in {elapsed}s")
        return result
    except Exception as e:
        print(f"❌ {name} FAILED: {e}")
        raise e


# ==============================
# FRONTEND JSON GENERATOR
# ==============================

def generate_frontend_json():
    print("\n🔹 Generating Frontend JSON...")
    frontend_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "frontend",
        "generate_json.py"
    )
    if not os.path.exists(frontend_path):
        raise FileNotFoundError(f"Missing generate_json script: {frontend_path}")

    spec = importlib.util.spec_from_file_location("generate_json", frontend_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.generate_json()
    print("✅ Frontend JSON generated successfully")


# ==============================
# PIPELINE MODES
# ==============================

def run_full_pipeline(skip_fetch: bool = False, skip_train: bool = False, export_json: bool = False):
    print("=" * 60)
    print("🚀 STARTING FULL PRODUCTION PIPELINE")
    print("=" * 60)

    if not skip_fetch:
        run_step("Build Play-by-Play Dataset", build_dataset)
    else:
        print("\n🔹 [1/11] Skipping Play-by-Play API fetch (--skip-fetch)...")

    run_step("Compute Player Ratings", compute_player_ratings)
    run_step("Add Game Dates", add_game_dates)
    run_step("Build Features", build_features)

    if not skip_train:
        run_step("Train Model", train_model)
    else:
        print("\n🔹 [5/11] Skipping Model Training (--skip-train)...")

    run_step("Predict Ratings", predict_ratings)
    run_step("Apply Injury Status", apply_injury_status)
    run_step("Get Next Day Games", get_next_day_games)
    run_step("Build Frontend Dataset", build_frontend_dataset)
    if export_json:
        run_step("Generate Frontend JSON", generate_frontend_json)
    else:
        print("\n🔹 Skipping legacy Frontend JSON generation (FastAPI serves parquet directly)...")
    run_step("Build Team Rankings", build_team_rankings)

    print("\n" + "=" * 60)
    print("🎉 FULL PIPELINE EXECUTION COMPLETE")
    print("=" * 60)


def run_training_pipeline():
    print("=" * 60)
    print("🚀 STARTING TRAINING-ONLY PIPELINE")
    print("=" * 60)

    run_step("Compute Player Ratings", compute_player_ratings)
    run_step("Add Game Dates", add_game_dates)
    run_step("Build Features", build_features)
    run_step("Train Model", train_model)

    print("\n" + "=" * 60)
    print("🎉 TRAINING PIPELINE COMPLETE")
    print("=" * 60)


def run_prediction_pipeline(export_json: bool = False):
    print("=" * 60)
    print("🚀 STARTING PREDICTION-ONLY PIPELINE")
    print("=" * 60)

    run_step("Predict Ratings", predict_ratings)
    run_step("Apply Injury Status", apply_injury_status)
    run_step("Get Next Day Games", get_next_day_games)
    run_step("Build Frontend Dataset", build_frontend_dataset)
    if export_json:
        run_step("Generate Frontend JSON", generate_frontend_json)
    else:
        print("\n🔹 Skipping legacy Frontend JSON generation (FastAPI serves parquet directly)...")
    run_step("Build Team Rankings", build_team_rankings)

    print("\n" + "=" * 60)
    print("🎉 PREDICTION PIPELINE COMPLETE")
    print("=" * 60)


# ==============================
# CLI PARSER & ENTRYPOINT
# ==============================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="NBA Player Performance Prediction Pipeline Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "train", "predict"],
        default="full",
        help="Pipeline execution mode: 'full' (all stages), 'train' (feature prep & training), or 'predict' (inference & frontend JSONs)"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip historical play-by-play API fetch (applies to 'full' mode)"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model re-training (applies to 'full' mode)"
    )
    parser.add_argument(
        "--export-json",
        action="store_true",
        help="Export static JSON files to lineup-builder/public (legacy compatibility)"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.mode == "full":
        kwargs = {"skip_fetch": args.skip_fetch, "skip_train": args.skip_train}
        if getattr(args, "export_json", False):
            kwargs["export_json"] = True
        run_full_pipeline(**kwargs)
    elif args.mode == "train":
        run_training_pipeline()
    elif args.mode == "predict":
        kwargs = {}
        if getattr(args, "export_json", False):
            kwargs["export_json"] = True
        run_prediction_pipeline(**kwargs)


if __name__ == "__main__":
    main()