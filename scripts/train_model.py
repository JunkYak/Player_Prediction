import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ==============================
# LOAD DATA
# ==============================

df = pd.read_parquet("data/player_features.parquet")

print("Dataset loaded")
print("Rows:", len(df))

# ==============================
# FEATURES / TARGET
# ==============================

features = [
    "last_game",
    "last3_avg",
    "last5_avg",
    "last7_avg"
]

X = df[features]
y = df["rating"]

# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

# ==============================
# MODEL
# ==============================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

print("Training model...")

model.fit(X_train, y_train)

# ==============================
# EVALUATION
# ==============================

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\nModel Performance")
print("MAE:", round(mae, 3))
print("R2:", round(r2, 3))

# ==============================
# SAVE MODEL
# ==============================

joblib.dump(model, "models/rating_model.pkl")

print("\nModel saved to models/rating_model.pkl")

# ==============================
# SAMPLE PREDICTIONS
# ==============================

sample = X_test.copy()
sample["actual"] = y_test
sample["predicted"] = preds

print("\nSample predictions:")
print(sample.head(10))