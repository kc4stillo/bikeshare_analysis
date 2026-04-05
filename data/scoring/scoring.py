# %%
import re

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

curr_stat = df.iloc[:72].copy()
projected_stat_df = df.iloc[72:].copy()

# %%
trips = pd.read_csv("../../data/raw/scoring/tips_per_station.csv")

# --- make copies ---
scores = curr_stat.copy()
trips_df = trips.copy()


# --- basic cleaning function ---
def clean_station_name(s):
    if pd.isna(s):
        return s
    s = str(s).strip()
    s = s.replace("\t", "")
    s = " ".join(s.split())  # remove weird extra spaces
    return s


# --- clean names first ---
scores["name_clean"] = scores["name"].apply(clean_station_name)
trips_df["name_clean"] = trips_df["name"].apply(clean_station_name)

# --- drop rows from trips based on your notes ---
drop_names = [
    "30th/Whitis",
    "E 3rd/Trinity",
    "E 4th/Neches @ Downtown Station",
    "E 5th/Shady @ Eastside Bus Plaza",
    "E 6th/Robert T. Martinez",
    "E 7th/Congress",
    "E 7th/Pleasant Valley",
    "E 8th/San Jacinto",
    "E Cesar Chavez/Pleasant Valley",
    "E. 7th/Congress",
    "Lakeshore/Austin Hostel",
    "TEST-LucJ",
    "W 4th/Guadalupe @ Republic Square",
    "Warehouse Station",
    "Webberville/Northwestern",
]

drop_names = [clean_station_name(x) for x in drop_names]
trips_df = trips_df[~trips_df["name_clean"].isin(drop_names)].copy()

# --- manual fixes for mismatched station names ---
# key = name in trips
# value = matching name in curr_stat
name_map = {
    "W 6th/Congress": "W 7th/Congress (W 6th/Congress)",
    "E 12th/San Jacinto @ State Capitol Visitors G": "E 12th/San Jacinto @ State Cap Visitors Garage",
}

trips_df["name_clean"] = trips_df["name_clean"].replace(name_map)

# --- merge ---
merged = scores.merge(
    trips_df,
    on="name_clean",
    how="outer",
    suffixes=("_orig", "_trips"),
)
# %%
merged.drop(
    [
        "name_trips",
        "name_clean",
        "total_checkouts",
        "trips_per_dock",
        "trips_per_dock_day",
        "active_date",
    ],
    inplace=True,
    axis=1,
)

merged["trips_per_dock"] = merged["trips"] / merged["total_docks"]

merged = merged.rename(columns={"name_orig": "name"})

# reorder columns into a cleaner order
col_order = [
    # station identity
    "id",
    "name",
    "district",
    # station capacity / metadata
    # usage metrics
    "trips",
    "total_docks",
    "trips_per_dock",
    "ebs_station",
    "checkouts_rank_per_day",
    # overall score
    # score components
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

merged = (
    merged[col_order]
    .sort_values("trips_per_dock", ascending=False)
    .reset_index(drop=True)
)

merged.head()
# %%
ut_names = [
    "Dean Keeton/Park Place",
    "Dean Keeton/Robert Dedman Dr",
    "Dean Keeton/Speedway",
    "Dean Keeton/Whitis",
    "E 21st/Speedway @ PCL",
    "E 23rd/San Jacinto @ DKR Stadium",
    "Guadalupe/West Mall @ University Co-op",
    "W 21st/Guadalupe",
    "W 21st/University",
    "W 22.5/Rio Grande",
    "W 22nd/Pearl",
    "W 23rd/San Gabriel",
    "W 26th/Nueces",
    "W 28th/Rio Grande",
]

merged["is_ut"] = merged["name"].isin(ut_names).astype(int)

# %%
merged.to_csv("../../data/cleaned/scoring/current_stations.csv", index=False)
