# %% [markdown]
# # CapMetro Bikeshare – Clean names, join scoring + coords, and manually patch missing coordinates

# %%
import re

import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# %%
DIRECTION_WORDS = r"(?:east|west|north|south|e|w|n|s)\.?"

# tokens that should NOT become part of the 2-street key
LANDMARK_WORDS = {
    "station",
    "parking",
    "garage",
    "visitors",
    "visitor",
    "capitol",
    "capitol station",
    "museum",
    "bullock",
    "convention",
    "center",
    "city",
    "hall",
    "library",
    "lbj",
    "bridge",
    "pedestrian",
    "mopac",
    "auditorium",
    "palmer",
    "hq",
    "capital",
    "metro",
    "square",
    "republic",
    "park",
    "pease",
    "boardwalk",
    "west",
    "fairmont",
    "hostel",
    "victory",
    "grill",
    "acc",
    "ut",
    "mall",
    "the",
}


# %%
def _is_street_like(tok: str) -> bool:
    """
    Heuristic: keep things that look like streets:
    - contains a digit (6, 11, 22.5, etc)
    - or is a normal street name (letters, maybe 1-2 words)
    """
    if re.search(r"\d", tok):
        return True
    return bool(re.fullmatch(r"[a-z]+(?: [a-z]+){0,2}", tok))


def normalize_kiosk_name_v3(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x)

    # whitespace + lowercase
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[\t\r\n]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()

    # drop junk rows
    if re.search(r"(projected rubric score|station status|office/main/shop/repair)", s):
        return ""

    # remove parenthetical alternates
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)

    # normalize 22nd 1/2 -> 22.5
    s = re.sub(r"\b(\d+)(st|nd|rd|th)\s*1/2\b", r"\1.5", s)

    # ordinal -> number
    s = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", s)

    # remove road-type words
    s = re.sub(
        r"\b(street|st|avenue|ave|boulevard|blvd|road|rd|drive|dr|lane|ln|trail|trl)\b\.?",
        "",
        s,
    )

    # remove direction words (W/E/N/S)
    s = re.sub(rf"\b{DIRECTION_WORDS}\b", "", s)

    # unify separators to "/"
    s = re.sub(r"\s*(?:&|@| at | and |/)\s*", "/", s)
    s = re.sub(r"\s*-\s*", "/", s)

    # keep only letters/numbers/slash/spaces
    s = re.sub(r"[^a-z0-9/ ]+", "", s)

    # normalize spaces around slashes
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"/{2,}", "/", s).strip("/").strip()
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return ""

    parts = [p.strip() for p in s.split("/") if p.strip()]

    cleaned_parts = []
    for p in parts:
        words = [w for w in p.split() if w not in LANDMARK_WORDS]
        p2 = " ".join(words).strip()
        if p2:
            cleaned_parts.append(p2)

    street_candidates = [p for p in cleaned_parts if _is_street_like(p)]

    # If we have >=2, build a sorted intersection key
    if len(street_candidates) >= 2:
        a, b = street_candidates[0], street_candidates[1]
        return "/".join(sorted([a, b]))

    # Otherwise fall back to a place key
    if cleaned_parts:
        return cleaned_parts[0]

    return ""


# %%
# --------- load data ---------
file_path = "../raw/scoring/curr_station_rubric.xlsx"
scores = pd.read_excel(file_path, header=2)

coords_file_path = "../raw/scoring/kiosk_locations.csv"
coords_df = pd.read_csv(coords_file_path)
coords_df = coords_df[coords_df["Kiosk Status"].astype(str).str.lower() == "active"]

# %%
# --------- create name_clean in both tables ---------
coords_df["name_clean"] = coords_df["Kiosk Name"].map(normalize_kiosk_name_v3)
scores["name_clean"] = scores["name"].map(normalize_kiosk_name_v3)

coords = coords_df[coords_df["name_clean"].ne("")].copy()
scores = scores[scores["name_clean"].ne("")].copy()

# %%
# --------- join on cleaned key ---------
joined = scores.merge(
    coords, on="name_clean", how="left", suffixes=("_scores", "_coords")
)

loc = joined["Location"].astype("string")

# extract two numbers (handles negatives + decimals) into two new columns
extracted = loc.str.extract(r"\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)")

joined["lat"] = pd.to_numeric(extracted[0], errors="coerce")
joined["lon"] = pd.to_numeric(extracted[1], errors="coerce")

# %%
# --------- find score keys missing from coords ---------
coords_keys = pd.Index(coords["name_clean"].dropna().unique())
scores_keys = pd.Index(scores["name_clean"].dropna().unique())

in_scores_not_coords = scores_keys.difference(coords_keys).sort_values()
print(f"In scores, not in coords: {len(in_scores_not_coords)}")
print(in_scores_not_coords.tolist())

# %%
# --------- manual coordinate patches (lat/lon) for missing keys ---------
manual_coords = pd.DataFrame(
    [
        ("1/riverside", 30.259384, -97.749726),
        ("11/waller", 30.26899800040119, -97.72843433423911),
        ("12/san jacinto", 30.273499, -97.738097),
        ("30/whitis", 30.295427, -97.739347),
        ("5/neches", 30.265843991099903, -97.73891781267969),
        ("6/chicon", 30.259718, -97.723198),
        ("7/congress", 30.26822, -97.74285),
        ("atlanta/veterans", 30.274475, -97.769892),
        ("azie morton/barton springs", 30.261881964956064, -97.76897665654796),
        ("barton springs/bouldin", 30.25966, -97.753445),
        ("cesar chavez/pleasant valley", 30.252951, -97.712467),
        ("dean keeton/place", 30.28931, -97.733037),
        ("dean keeton/robert dedman", 30.28785, -97.728541),
        ("electric/pfluger ped", 30.267064, -97.75482),
        ("guadalupe/university co", 30.285664, -97.741792),
        ("lady bird/lakeshore", 30.24478312140979, -97.72319224423872),
        ("neal/webberville", 30.267506, -97.707997),
        ("northwestern/webberville", 30.263061, -97.713433),
        ("one texas", 30.257653, -97.74898),
    ],
    columns=["name_clean", "lat_manual", "lon_manual"],
)

# %%
# --------- patch joined with manual coords (fill only when missing) ---------
# Try to detect which columns in your coords file are latitude/longitude.
lat_candidates = [c for c in joined.columns if c.lower() in {"lat", "latitude", "y"}]
lon_candidates = [
    c for c in joined.columns if c.lower() in {"lon", "lng", "longitude", "x"}
]

if not lat_candidates or not lon_candidates:
    raise KeyError(
        "Could not find latitude/longitude columns in `joined`.\n"
        f"Found lat candidates: {lat_candidates}\n"
        f"Found lon candidates: {lon_candidates}\n"
        "Rename the coords columns to `lat`/`lon` (or adjust the detection logic)."
    )

LAT_COL = lat_candidates[0]
LON_COL = lon_candidates[0]

joined = joined.merge(manual_coords, on="name_clean", how="left")

joined[LAT_COL] = joined[LAT_COL].where(joined[LAT_COL].notna(), joined["lat_manual"])
joined[LON_COL] = joined[LON_COL].where(joined[LON_COL].notna(), joined["lon_manual"])

joined = joined.drop(columns=["lat_manual", "lon_manual"])

# %%
# --------- final checks: what still has missing coords? ---------
missing_after_patch = joined[joined[LAT_COL].isna() | joined[LON_COL].isna()][
    ["name_clean"]
].drop_duplicates()

print("Still missing coords after manual patch:", len(missing_after_patch))
print(missing_after_patch["name_clean"].sort_values().tolist())

# %%
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# %%
# fixing ut stations
# Stations considered "on UT property"
ut_stations = {
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
}

# Add binary column: 1 if station name is in the UT set, else 0
joined["on_UT"] = joined["name"].isin(ut_stations).astype(int)


# %%
# drop checkout rankings
joined = joined.drop("Checkouts Rankings; per day >5=3; 2-5=2; <1=1 ", axis=1)


out_path = "../raw/scoring/raw_scores_with_coords.csv"
joined.to_csv(out_path, index=False)
print("Saved:", out_path)
