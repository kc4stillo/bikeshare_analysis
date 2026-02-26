import re

import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


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


def _is_street_like(tok: str) -> bool:
    """
    Heuristic: keep things that look like streets:
    - contains a digit (6, 11, 22.5, etc)
    - or is a normal street name (letters, maybe 1-2 words)
    """
    if re.search(r"\d", tok):
        return True
    # common street-name patterns: "guadalupe", "rio grande", "san antonio", etc.
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

    # split multiword parts into “tokens” but keep phrases like "rio grande" intact
    # (we’ll still treat the whole part as a candidate)
    cleaned_parts = []
    for p in parts:
        # remove landmark-y words inside parts
        words = [w for w in p.split() if w not in LANDMARK_WORDS]
        p2 = " ".join(words).strip()
        if p2:
            cleaned_parts.append(p2)

    # Now choose the 2 best “street-like” candidates
    street_candidates = [p for p in cleaned_parts if _is_street_like(p)]

    # If we have >=2, build a sorted intersection key
    if len(street_candidates) >= 2:
        a, b = street_candidates[0], street_candidates[1]
        return "/".join(sorted([a, b]))

    # Otherwise fall back to a place key
    if cleaned_parts:
        return cleaned_parts[0]

    return ""


file_path = "../raw/scoring/curr_station_rubric.xlsx"
scores = pd.read_excel(file_path, header=2)

coords_file_path = "../raw/scoring/kiosk_locations.csv"
coords_df = pd.read_csv(coords_file_path)
coords_df = coords_df[coords_df["Kiosk Status"] == "active"]

# --------- apply to BOTH dataframes/columns ---------
coords_df["name_clean"] = coords_df["Kiosk Name"].map(normalize_kiosk_name_v3)
scores["name_clean"] = scores["name"].map(normalize_kiosk_name_v3)

# optional: drop empties (non-stations / junk rows)
coords = coords_df[coords_df["name_clean"].ne("")].copy()
scores = scores[scores["name_clean"].ne("")].copy()

# --------- join on the cleaned key ---------
joined = scores.merge(coords, on="name_clean", how="left", suffixes=("_coords", "_raw"))

coords_keys = pd.Index(coords["name_clean"].dropna().unique())
scores_keys = pd.Index(scores["name_clean"].dropna().unique())

# in scores but not in coords
in_scores_not_coords = scores_keys.difference(coords_keys).sort_values()

print(f"In scores, not in coords: {len(in_scores_not_coords)}")
display(pd.DataFrame({"name_clean": in_scores_not_coords}).reset_index(drop=True))

# 	name_clean
# 0	1/riverside
# 1	11/waller
# 2	12/san jacinto
# 3	30/whitis
# 4	5/neches
# 5	6/chicon
# 6	7/congress
# 7	7/pleasant valley
# 8	8/trinity
# 9	atlanta/veterans
# 10	azie morton/barton springs
# 11	barton springs/bouldin
# 12	cesar chavez/pleasant valley
# 13	dean keeton/place
# 14	dean keeton/robert dedman
# 15	electric/pfluger ped
# 16	guadalupe/university co
# 17	lady bird/lakeshore
# 18	neal/webberville
# 19	northwestern/webberville
# 20	one texas

# missing coords

# 0	1/riverside = 30.259384	-97.749726
# 1	11/waller = 30.26899800040119, -97.72843433423911
# 2	12/san jacinto = 30.273499	-97.738097
# 3	30/whitis = 30.295427	-97.739347
# 4	5/neches = 30.265843991099903, -97.73891781267969
# 5	6/chicon = 30.259718	-97.723198
# 6	7/congress = (30.26822°, -97.74285°)
# 7	7/pleasant valley = (30.26025°, -97.71002°) DELETE THIS ONE ACTUALLY
# 8	8/trinity = ???
# 9	atlanta/veterans = 30.274475	-97.769892
# 10	azie morton/barton springs = 30.261881964956064, -97.76897665654796
# 11	barton springs/bouldin = 30.25966	-97.753445
# 12	cesar chavez/pleasant valley = 30.252951	-97.712467
# 13	dean keeton/place = 30.28931	-97.733037
# 14	dean keeton/robert dedman = 30.28785	-97.728541
# 15	electric/pfluger ped = 30.267064	-97.75482
# 16	guadalupe/university co = 30.285664	-97.741792
# 17	lady bird/lakeshore = 30.24478312140979, -97.72319224423872
# 18	neal/webberville = 30.267506	-97.707997
# 19	northwestern/webberville = 30.263061	-97.713433
# 20	one texas= 30.257653	-97.74898
