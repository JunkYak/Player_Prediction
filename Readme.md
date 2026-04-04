# 🏀 Player Performance Prediction System

A data-driven system that processes NBA play-by-play data to generate custom player performance ratings and predict future performances using historical trends.

---

## 🚀 Overview

This project builds a full pipeline to:

- Extract play-by-play data from NBA API  
- Compute **custom player performance ratings**  
- Generate **historical features (form-based)**  
- Train a machine learning model to predict **next-game performance**  
- Provide **team-wise player rankings** for draft-style decision making  

---

## ⚙️ Pipeline Architecture

Play-by-Play Data
↓
Rating Engine (Custom Formula)
↓
Player Game Ratings
↓
Feature Engineering (last_game, rolling avg)
↓
Model Training (Random Forest)
↓
Prediction + Ranking


---

## 🧠 Core Concepts

### 1. Custom Rating System

Each player’s performance is calculated using:

- Shot quality & difficulty  
- Assists / unassisted scoring  
- Defensive actions (steals, blocks, rebounds)  
- Context (lead change, clutch moments)  
- Clutch multiplier (4th quarter weighting)  

---

### 2. Feature Engineering

For each player-game:

- `last_game` → previous game rating  
- `last3_avg` → average of last 3 games  
- `last5_avg` → average of last 5 games  
- `last7_avg` → average of last 7 games  

👉 These represent **player form**

---

### 3. Model

- Model: **Random Forest Regressor**  
- Input: historical form features  
- Output: **predicted rating for next game**  

---

## 📂 Project Structure

---

## 🧠 Core Concepts

### 1. Custom Rating System

Each player’s performance is calculated using:

- Shot quality & difficulty  
- Assists / unassisted scoring  
- Defensive actions (steals, blocks, rebounds)  
- Context (lead change, clutch moments)  
- Clutch multiplier (4th quarter weighting)  

---

### 2. Feature Engineering

For each player-game:

- `last_game` → previous game rating  
- `last3_avg` → average of last 3 games  
- `last5_avg` → average of last 5 games  
- `last7_avg` → average of last 7 games  

👉 These represent **player form**

---

### 3. Model

- Model: **Random Forest Regressor**  
- Input: historical form features  
- Output: **predicted rating for next game**  

---

## 📂 Project Structure
Player_Prediction/
│
├── data/
│ ├── play_by_play.parquet
│ ├── player_game_ratings.parquet
│ ├── player_game_ratings_with_dates.parquet
│ ├── player_features.parquet
│ └── predicted_ratings.parquet
│
├── models/
│ └── rating_model.pkl
│
├── scripts/
│ ├── build_play_by_play_dataset.py
│ ├── compute_player_ratings.py
│ ├── add_game_dates.py
│ ├── build_features.py
│ ├── train_model.py
│ └── test_predictions.py
│
├── rating_engine/
│ └── formula.py
│
└── README.md

## 📊 Output Example

playerName teamId gameId predicted_rating last_game

Dončić 1610612747 0022500974 4.82 8.66
Jokić 1610612743 0022500974 4.50 6.10
Booker 1610612756 0022500963 4.29 7.58

## 🛠️ Setup

### 1. Install dependencies


pip install pandas numpy scikit-learn nba_api pyarrow joblib


---

## ▶️ How to Run

### Step 1: Build play-by-play dataset


python scripts/build_play_by_play_dataset.py


### Step 2: Compute player ratings


python scripts/compute_player_ratings.py

### Step 4: Build features


python scripts/build_features.py


### Step 5: Train model


python scripts/train_model.py


### Step 6: Generate predictions


python scripts/test_predictions.py

## 🎯 Use Case

This system is designed for:

- Fantasy draft decision support  
- Identifying undervalued players  
- Comparing player form across teams  
- Building a team-wise ranking interface  

---

## ⚠️ Current Limitations

- Uses only **form-based features**
- Does NOT include:
  - Minutes played  
  - Injury status  
  - Opponent strength  
  - Home/Away context  

👉 Model predicts **trend**, not full-context performance  

---

## 🔮 Future Improvements

Planned enhancements:

- Add minutes + usage features  
- Injury tracking integration  
- Opponent defense strength  
- Home vs away splits  
- Back-to-back fatigue modeling  
- Upgrade model to XGBoost / LightGBM  
- Build UI (Streamlit / Web App)  

---

## 🧠 Key Insight

This is not just a prediction model — it’s a:

> **Player form + momentum tracking system**

Which is often more useful than raw stats in draft scenarios.

---

## 📌 Status

- ✅ End-to-end pipeline complete  
- ✅ Predictions working  
- ⚠️ Feature set minimal (needs expansion)  
- 🚀 Ready for prototype UI / product layer 