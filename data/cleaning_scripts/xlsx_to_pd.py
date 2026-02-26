# %%
import re
from pathlib import Path

import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# %%
def to_snake(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s


def clean_station_rubric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize column labels
    df.columns = [str(c).strip() for c in df.columns]

    # 1) Standardize station name

    if "name" in df.columns:
        df["name"] = (
            df["name"]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace("/", " and ", regex=False)
            .str.replace("@", " at ", regex=False)
            .str.replace(" ", "_", regex=False)
        )

    # 2) Rename long columns
    rename_map = {
        "Active Date": "active_date",
        "Districts": "district",
        "total Checkouts": "total_checkouts",
        "total Docks": "total_docks",
        "trips per dock": "trips_per_dock",
        "trips per dock/day": "trips_per_dock_day",
        "EBS STATION": "ebs_station",
        "Checkouts Rankings; per day >5=3; 2-5=2; <1=1": "checkouts_rank_per_day",
        "Co-locate to Transit (at transit =3; <1/4 mi = 2; >1/4 mi = 1)": "transit_access_score",
        "Access to Jobs (Major employment hubs)  (1/4 mi = 3; 1/2 mi = 2; >1/2 = 1)": "jobs_access_score",
        "Access to Households  (1/4 mi = 3; 1/2 mi = 2;  >1/2 = 1)": "households_access_score",
        "Access to low income residents (1/4 mi = 3; 1/2 mi = 2; >1/2 = 1)": "low_income_access_score",
        "Access to Public amenities (libraries, schools, Rec Centers, parks)  (1/4 mi = 3; 1/2 mi = 2; >1/2 = 1)": "public_amenities_access_score",
        "Bikeable infrastructure (rider saftey)  (1/4 mi = 3; 1/2 mi = 2; >1/2 = 1)": "bike_infra_score",
        "Access to retail or entertainment  (1/4 mi = 3; 1/2 mi = 2; >1/2 = 1)": "retail_entertainment_access_score",
        "Access to existing Bikeshare footprint - 1/4 mi = 3; 1/2 mi = 2; >1/2 = 1": "existing_bikeshare_access_score",
        "Total Score": "total_score",
    }

    df = df.rename(columns=rename_map)
    df = df.rename(columns={c: to_snake(c) for c in df.columns})

    # 3) Fix datatypes
    if "id" in df.columns:
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
        df = df[df["id"].notna()].copy()  # drop rows where id is missing

    if "district" in df.columns:
        df["district"] = pd.to_numeric(df["district"], errors="coerce").astype("Int64")

    if "total_docks" in df.columns:
        df["total_docks"] = pd.to_numeric(df["total_docks"], errors="coerce").astype(
            "Int64"
        )
    if "total_checkouts" in df.columns:
        df["total_checkouts"] = pd.to_numeric(
            df["total_checkouts"], errors="coerce"
        ).astype("Int64")

    numeric_cols = [
        "total_checkouts",
        "trips_per_dock",
        "trips_per_dock_day",
        "checkouts_rank_per_day",
        "transit_access_score",
        "jobs_access_score",
        "households_access_score",
        "low_income_access_score",
        "public_amenities_access_score",
        "bike_infra_score",
        "retail_entertainment_access_score",
        "existing_bikeshare_access_score",
        "total_score",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "active_date" in df.columns:
        df["active_date"] = pd.to_datetime(df["active_date"], errors="coerce")

    # 4) Clean EBS column
    if "ebs_station" in df.columns:
        df["ebs_station"] = df["ebs_station"].astype("string").str.strip()

        df["ebs_station"] = df["ebs_station"].replace({"✓": 1, "<NA>": 0})

        # Convert remaining missing to 0
        df["ebs_station"] = df["ebs_station"].fillna(0).astype(int)

    # 5) Replace <NA> strings everywhere else with np.nan

    return df


# %%
file_path = "../raw/scoring/curr_station_rubric.xlsx"
raw_df = pd.read_excel(file_path, header=2)

df = clean_station_rubric(raw_df)

current_stat_df = df.iloc[:72].copy()
projected_stat_df = df.iloc[72:].copy()

# %%
out_dir = Path("../cleaned/scoring")
out_dir.mkdir(parents=True, exist_ok=True)

current_stat_df.to_csv(out_dir / "current_stations_cleaned.csv", index=False)
projected_stat_df.to_csv(out_dir / "projected_stations_cleaned.csv", index=False)
