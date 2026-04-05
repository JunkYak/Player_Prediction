import pandas as pd
import os
import json


# ==============================
# PATH CONFIG
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "lineup-builder", "public")

HOME_FILE = os.path.join(DATA_DIR, "frontend_home.parquet")
ALL_FILE = os.path.join(DATA_DIR, "frontend_all.parquet")

OUTPUT_HOME = os.path.join(OUTPUT_DIR, "frontend_home.json")
OUTPUT_ALL = os.path.join(OUTPUT_DIR, "frontend_all.json")


# ==============================
# ENSURE OUTPUT DIR EXISTS
# ==============================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# CONVERT FUNCTION
# ==============================

def convert_parquet_to_json(input_path, output_path):

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing file: {input_path}")

    df = pd.read_parquet(input_path)

    if df.empty:
        raise Exception(f"{input_path} is empty")

    # ==============================
    # FIX: Convert datetime columns
    # ==============================

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)

    # Also ensure no NaNs break JSON
    df = df.fillna("")

    records = df.to_dict(orient="records")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Saved → {output_path}")
    print(f"Records: {len(records)}")


# ==============================
# MAIN
# ==============================

def generate_json():

    print("\n🔹 Generating frontend JSON files...")

    convert_parquet_to_json(HOME_FILE, OUTPUT_HOME)
    convert_parquet_to_json(ALL_FILE, OUTPUT_ALL)

    print("\n✅ Frontend JSON ready")


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    generate_json()