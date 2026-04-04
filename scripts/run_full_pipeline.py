import argparse
import sys
import os

from build_play_by_play_dataset import build_dataset
from compute_player_ratings import compute_player_ratings
from add_game_dates import add_game_dates
from build_features import build_features
from train_model import train_model
from predict_ratings import predict_ratings
from get_next_day_games import get_next_day_games
from build_frontend_dataset import build_frontend_dataset

def main():
    parser = argparse.ArgumentParser(description="NBA Prediction Pipeline Executer")
    parser.add_argument('--skip-fetch', action='store_true', help="Skip fetching historical play-by-play data (saves ~4 minutes)")
    parser.add_argument('--skip-train', action='store_true', help="Skip re-training the Random Forest model")
    parser.add_argument('--fast', action='store_true', help="Only run predictions on next day's games (skips fetching, computing all ratings, and training)")
    args = parser.parse_args()

    print("=" * 40)
    print("🚀 STARTING NBA PREDICTION PIPELINE")
    print("=" * 40)

    try:
        if not args.fast and not args.skip_fetch:
            print("\n[1/8] Building Play-by-Play Dataset...")
            build_dataset()
        else:
            print("\n[1/8] Skipping Raw Data Fetch...")

        if not args.fast:
            print("\n[2/8] Computing Player Ratings...")
            compute_player_ratings()

            print("\n[3/8] Adding Game Dates...")
            add_game_dates()

            print("\n[4/8] Building Features...")
            build_features()
        else:
            print("\n[2-4/8] Skipping Feature Computation (Fast Mode)...")

        if not args.fast and not args.skip_train:
            print("\n[5/8] Training Random Forest Mode...")
            train_model()
        else:
            print("\n[5/8] Skipping Model Training...")

        print("\n[6/8] Generating Upcoming Predictions...")
        predict_ratings()

        print("\n[7/8] Aligning with Tomorrow's Games...")
        get_next_day_games()

        print("\n[8/8] Rendering Frontend JSONs...")
        build_frontend_dataset()

        # Execute legacy json builder script
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "frontend"))
        import importlib.util
        spec = importlib.util.spec_from_file_location("generate_json", os.path.join(os.path.dirname(__file__), "..", "frontend", "generate_json.py"))
        gen_json = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gen_json)

        print("\n✅ PIPELINE EXECUTION COMPLETE! READY FOR DASHBOARD.")

    except Exception as e:
        print(f"\n❌ PIPELINE TERMINATED DUE TO ERROR: {e}")

if __name__ == "__main__":
    main()
