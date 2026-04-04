# 🏀 NBA Player Performance Prediction System - Project Context

This document provides a highly detailed technical overview of the "Player Prediction" project. It is intended to serve as a comprehensive onboarding guide for anyone starting work on this codebase, explaining the architecture, data flow, scripts, and underlying logic.

---

## 1. 🚀 Project Overview

The Player Performance Prediction System is a data-driven pipeline that processes NBA play-by-play data to generate custom performance ratings for players. It then uses historical performance trends (form) to train a machine learning model capable of predicting future player performances. Finally, the system visualizes these predictions in a team-wise ranking interface, primarily to support fantasy draft decisions.

### Core Objectives
- Process raw NBA play-by-play data.
- Compute non-standard, custom player performance ratings based on context, shot difficulty, and clutch moments.
- Engineer historical form-based features (e.g., rolling averages).
- Train a Random Forest model to predict the next game's rating.
- Expose the predictions for a visually appealing, responsive web dashboard.

---

## 2. 📂 Directory Structure

The project root `d:\Projects\Player Prediction\` is logically divided into several key directories:

- `data/` - Contains raw, intermediate, and final processed data. Primarily stores `.parquet` files for efficiency (e.g., `play_by_play.parquet`, `predicted_ratings.parquet`, frontend dataset files).
- `models/` - Stores the trained ML models (e.g., `rating_model.pkl`).
- `scripts/` - Contains the Python scripts forming the steps of the data ingestion, processing, and prediction pipeline.
- `rating_engine/` - Holds the core logic for the custom metric calculations (`formula.py`).
- `frontend/` - Contains the web application (HTML, CSS, JS) and scripts to generate JSON endpoints for the dashboard.
- `images/`, `misc/` - Auxiliary assets and miscellaneous project files.
- `ML_Project_Report.md / .tex`, `Model_Metrics_Notebook.ipynb` - Technical documentation and theoretical model evaluations.
- `Readme.md` - A high-level overview.

---

## 3. ⚙️ Data Flow & Pipeline Architecture

The entire project operates as a sequential pipeline. The master script orchestrating this is `scripts/run_full_pipeline.py`. The pipeline consists of the following steps, executed in order:

1. **`build_play_by_play_dataset.py`**: Fetches raw play-by-play data from the NBA API and converts it into a structural DataFrame, saved to `data/play_by_play.parquet`.
2. **`compute_player_ratings.py`**: Applies the custom rating engine formula to the play-by-play data to calculate each player's definitive rating for a specific game.
3. **`add_game_dates.py`**: Enriches the dataset by merging the game ratings with actual sequence/dates, ensuring chronologically accurate downstream processing.
4. **`build_features.py`**: Generates form-based historical features. Specifically, it computes:
    - `last_game`: The player's rating in their immediately preceding game.
    - `last3_avg`, `last5_avg`, `last7_avg`: Rolling averages over the last 3, 5, and 7 games, representing form and momentum.
5. **`train_model.py`**: Trains the machine learning model. It uses a **Random Forest Regressor** to learn the mapping from historical form features to the next game's rating. The trained model is saved to `models/rating_model.pkl`.
6. **`predict_ratings.py` / `test_predictions.py`**: Feeds current player states into the trained model to predict output ratings for upcoming games.
7. **`get_next_day_games.py`**: Fetches the NBA schedule to determine who is playing in the upcoming slate of games, to align predictions with reality.
8. **`build_frontend_dataset.py`**: Formats the final predictions and historical data into clean structures (`frontend_home.parquet`, `frontend_all.parquet`) suitable for the dashboard.

---

## 4. 🧠 The Custom Rating Engine (`rating_engine/formula.py`)

A unique aspect of this project is the custom rating evaluation (defined in `formula.py`). Standard stats (like raw points or assists) are not used directly. Instead, scores are derived via specific modifiers:

### A. Base Delta (`base_delta`)
Assigns granular value to every play based on specific textual descriptions and actions context:
- **Turnovers**: `-0.17`
- **Blocks/Steals**: `~0.18 - 0.19`
- **Shot Modifications**: Three-pointers, floaters, hook shots, step-backs, and fadeaways receive different fundamental weights. For instance, a step-back shot adds +`0.06` difficulty points, and deep threes (>27ft) get a bonus.
- **Assisted vs. Unassisted**: An unassisted basket is inherently worth more (+`0.03`) than an assisted one (`-0.04`).

### B. Context Delta (`context_delta`)
Evaluates the *importance* of the basket in relation to the prevailing score difference.
- Game-winners award a massive `+0.8` boost.
- Game-tying or lead-taking shots add substantial bonuses compared to blowout points.

### C. Clutch Multiplier (`clutch_multiplier`)
Magnifies the value of actions performed under pressure.
- Triggers strictly in the **4th Quarter**.
- For the final 120 seconds of the game, multipliers linearly increase to significantly reward "clutch" shots that save/win games.

---

## 5. 🖥️ Frontend Dashboard

Located in the `frontend/` directory, the interface consumes the predictions.

- **Stack**: Vanilla HTML/JS with custom CSS (designed around modern aesthetics, glassmorphic UI elements, dynamic interactions).
- **Data Ingestion**: A bridge script `generate_json.py` takes the pipelined `frontend_*.parquet` files and converts them into `.json` structures that the frontend (`app.js`) can asynchronously fetch and populate the DOM with.
- **Functionality**: Features team-wise data filtering and top-performer highlights based on the Random Forest predictions.

---

## 6. ⚠️ Current Limitations & Future Improvements

To immediately start contributing, it's essential to understand the current blind spots of the predictive system:

### Known Limitations
- The model exclusively relies on **Momentum/Form** (`last_n_avg`).
- It completely ignores critical real-world factors:
  - **Minutes Played & Usage Rate**: Assuming equal opportunities.
  - **Injuries**: No system is currently in place to remove injured players from the predicted roster.
  - **Opponent Strength & Context**: Matchup specific advantages, along with Home/Away fatigue and back-to-back schedules, are not considered.

### Future Roadmap
- Integration of injury API endpoints to prune unavailable players.
- Adding Usage Rate and Minutes as features to normalize the ratings.
- Opposing Defense metrics API merging.
- Consider attempting an upgrade from Random Forest to XGBoost or LightGBM for non-linear feature interactions when the dataset expands.
