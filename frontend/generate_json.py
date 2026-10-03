import pandas as pd
import os
import sys
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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

def convert_parquet_to_json(input_path: str, output_path: str):
    """
    Safely converts a parquet dataset to a JSON array.
    If the parquet is empty (e.g. 0 games scheduled tomorrow), writes [] to output.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing file: {input_path}")

    df = pd.read_parquet(input_path)

    if df.empty:
        records = []
    else:
        # Convert datetime columns to ISO string
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)

        # Ensure no NaNs break JSON serialization
        df = df.fillna("")
        records = df.to_dict(orient="records")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Saved → {output_path} ({len(records)} records)")


# ==============================
# MAIN
# ==============================

def generate_json():
    print("\n🔹 Generating frontend JSON files...")
    convert_parquet_to_json(HOME_FILE, OUTPUT_HOME)
    convert_parquet_to_json(ALL_FILE, OUTPUT_ALL)
    print("✅ Frontend JSON ready")


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    generate_json()