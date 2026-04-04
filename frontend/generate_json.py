import pandas as pd
import json
import os

# Define paths relative to the frontend directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

home_parquet_path = os.path.join(DATA_DIR, "frontend_home.parquet")
all_parquet_path = os.path.join(DATA_DIR, "frontend_all.parquet")

home_json_path = os.path.join(DATA_DIR, "frontend_home.json")
all_json_path = os.path.join(DATA_DIR, "frontend_all.json")

print(f"Reading {home_parquet_path}...")
try:
    df_home = pd.read_parquet(home_parquet_path)
    df_home.to_json(home_json_path, orient="records")
    print(f"Successfully generated {home_json_path}")
except Exception as e:
    print(f"Error processing frontend_home.parquet: {e}")

print(f"Reading {all_parquet_path}...")
try:
    df_all = pd.read_parquet(all_parquet_path)
    df_all.to_json(all_json_path, orient="records")
    print(f"Successfully generated {all_json_path}")
except Exception as e:
    print(f"Error processing frontend_all.parquet: {e}")

print("Done generating JSON files for the frontend.")
