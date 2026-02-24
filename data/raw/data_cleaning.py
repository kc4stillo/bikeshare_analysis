# %%
# TODO:
# standardize name column
# lowercase everything
# replace / with _and_
# replace @ with _at_
# standardize column names
# all lowercase
# shorten
# snake_case
# fix datatypes in columns
# add `added_docs` column from excel

# %%
import re

import pandas as pd

# %%
file_path = "curr_station_rubric.xlsx"
df = pd.read_excel(file_path, header=2)

# %%
# 1) standardize station name column
df["name"] = (
    df["name"]
    .astype("string")
    .str.strip()
    .str.lower()
    .str.replace("/", "_and_", regex=False)
    .str.replace("@", "_at_", regex=False)
)

# %%
# 2) standardize / shorten column names (snake_case)
#    We use an explicit rename map so you don't end up with 200-char headers.
rename_map = {
    "Active Date": "active_date",
    "Districts": "district",
    "total Checkouts": "total_checkouts",
    "total Docks": "total_docks",
    "trips per dock": "trips_per_dock",
    "trips per dock/day": "trips_per_dock_day",
    "EBS STATION": "ebs_station",
    "Checkouts Rankings; per day >5=3; 2-5=2; <1=1 ": "checkouts_rank_per_day",
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

# before renaming, strip weird whitespace (you have a trailing space in that checkouts header)
df.columns = [str(c).strip() for c in df.columns]
df = df.rename(columns=rename_map)


# OPTIONAL: if any columns weren't in the map, make them snake_case automatically
def to_snake(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]", "", s)  # drop punctuation
    s = re.sub(r"\s+", "_", s)  # spaces -> underscore
    s = re.sub(r"_+", "_", s)  # collapse underscores
    return s


df = df.rename(columns={c: to_snake(c) for c in df.columns})

# %%
# 3) fix datatypes

# id + district should be integer-like (allow missing)
if "id" in df.columns:
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")

if "district" in df.columns:
    df["district"] = pd.to_numeric(df["district"], errors="coerce").astype("Int64")

# total_docks is currently object; coerce to numeric
if "total_docks" in df.columns:
    df["total_docks"] = pd.to_numeric(df["total_docks"], errors="coerce").astype(
        "Int64"
    )

# numeric columns (scores + continuous metrics)
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

# active_date should be datetime (already is, but enforce)
if "active_date" in df.columns:
    df["active_date"] = pd.to_datetime(df["active_date"], errors="coerce")

# ebs_station: normalize to boolean-ish flag (optional)
if "ebs_station" in df.columns:
    df["ebs_station"] = (
        df["ebs_station"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA})
    )

# %%
df.dtypes
current = df.iloc[:72]
projected = df.iloc[74:81]


# %%
