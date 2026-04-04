import time

from scripts.build_play_by_play_dataset import build_dataset
from scripts.compute_player_ratings import compute_player_ratings
from scripts.add_game_dates import add_game_dates
from scripts.build_features import build_features
from scripts.train_model import train_model
from scripts.predict_ratings import predict_ratings
from scripts.team_rankings import build_team_rankings
from scripts.get_next_day_games import get_next_day_games


def run_step(name, func):
    print(f"\n🔹 {name}...")
    start = time.time()

    try:
        result = func()
        print(f"✅ {name} completed in {round(time.time() - start, 2)}s")
        return result
    except Exception as e:
        print(f"❌ {name} FAILED")
        print(e)
        raise e


def run_full_pipeline():
    run_step("Build Play-by-Play Dataset", build_dataset)
    run_step("Compute Player Ratings", compute_player_ratings)
    run_step("Add Game Dates", add_game_dates)
    run_step("Build Features", build_features)
    run_step("Train Model", train_model)
    run_step("Predict Ratings", predict_ratings)
    run_step("Get Next Day Games", get_next_day_games)
    run_step("Build Team Rankings", build_team_rankings)


def run_training_pipeline():
    run_step("Compute Player Ratings", compute_player_ratings)
    run_step("Add Game Dates", add_game_dates)
    run_step("Build Features", build_features)
    run_step("Train Model", train_model)


def run_prediction_pipeline():
    run_step("Predict Ratings", predict_ratings)
    run_step("Get Next Day Games", get_next_day_games)
    run_step("Build Team Rankings", build_team_rankings)


if __name__ == "__main__":

    print("\nChoose mode:")
    print("1 → FULL PIPELINE")
    print("2 → TRAIN ONLY")
    print("3 → PREDICT ONLY")

    choice = input("Enter choice: ")

    if choice == "1":
        run_full_pipeline()
    elif choice == "2":
        run_training_pipeline()
    elif choice == "3":
        run_prediction_pipeline()
    else:
        print("Invalid choice")