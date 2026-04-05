import pandas as pd
from nbainjuries import injury
from datetime import datetime, timedelta
import unicodedata


# ==============================
# HELPERS
# ==============================

def normalize_injury_name(name):
    if not isinstance(name, str):
        return None

    if "," not in name:
        name = name.lower().strip()
    else:
        last, first = name.split(",")
        name = f"{first.strip()} {last.strip()}".lower()

    # remove accents
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')

    return name


def normalize_name(name):
    if not isinstance(name, str):
        return None

    name = name.lower().strip()

    # remove accents
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')

    return name


# ==============================
# FETCH INJURY DATA
# ==============================

def fetch_injury_data():

    for days_back in range(0, 3):

        date = datetime.now() - timedelta(days=days_back)
        timestamp = datetime(date.year, date.month, date.day, 17, 30)

        try:
            df = injury.get_reportdata(timestamp, return_df=True)
            print(f"Using injury report at: {timestamp}")
            return df
        except:
            print(f"Failed for: {timestamp}")

    raise Exception("No valid injury report found")


# ==============================
# MAIN
# ==============================

def apply_injury_status():

    # LOAD
    pred = pd.read_parquet("data/predicted_ratings.parquet")
    print("Loaded predictions:", pred.shape)

    # FETCH INJURY DATA
    injury_df = fetch_injury_data()
    print("Fetched injuries:", injury_df.shape)

    # ==============================
    # NORMALIZE
    # ==============================

    injury_df["norm_name"] = injury_df["Player Name"].apply(normalize_injury_name)
    injury_df = injury_df[injury_df["norm_name"].notna()]

    pred["norm_name"] = pred["playerName"].apply(normalize_name)

    # ==============================
    # LAST NAME MATCHING (CORE FIX)
    # ==============================

    injury_df["last_name"] = injury_df["norm_name"].apply(
        lambda x: x.split()[-1] if isinstance(x, str) else None
    )

    pred["last_name"] = pred["norm_name"].apply(
        lambda x: x.split()[-1] if isinstance(x, str) else None
    )

    # ==============================
    # MERGE
    # ==============================

    merged = pred.merge(
        injury_df[["last_name", "Current Status"]],
        on="last_name",
        how="left"
    )

    # ==============================
    # STATUS
    # ==============================

    def simplify_status(s):
        if pd.isna(s):
            return "Active"

        s = s.lower()

        if "out" in s:
            return "Out"
        elif "questionable" in s:
            return "Questionable"
        elif "probable" in s:
            return "Probable"
        elif "available" in s:
            return "Active"
        else:
            return "Active"


    merged["status"] = merged["Current Status"].apply(simplify_status)
    merged = merged.drop(columns=["Current Status"])

    # ==============================
    # SAVE
    # ==============================

    merged.to_parquet(
        "data/predicted_ratings_with_status.parquet",
        index=False
    )

    print("\nStatus distribution:")
    print(merged["status"].value_counts())

    print("\nSaved → data/predicted_ratings_with_status.parquet")

    return merged


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    apply_injury_status()