# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", None)


# %%
# api shit
def api_lookup(category, year=2024, state_fips="48", county_fips="453"):
    url = (
        f"https://api.census.gov/data/{year}/acs/acs5"
        f"?get=NAME,group({category})"
        f"&for=block%20group:*"
        f"&in=state:{state_fips}+county:{county_fips}"
    )

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])

    # build block-group GEOID
    df["GEOID"] = df["state"] + df["county"] + df["tract"] + df["block group"]

    # drop duplicate name columns (if exists)
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def clean_attribute(col):
    bad_values = {
        -666666666,
        -222222222,
        "-666666666",
        "666666666",
        "666666666.0",
        "-222222222",
        "-666666666.0",
        -666666666.0,
        "-",
        "**",
        "***",
        "*****",
        "N",
        "null",
        "None",
        "",
    }

    return col.replace(list(bad_values), np.nan).pipe(pd.to_numeric, errors="coerce")


# %%
# AGE BY BLOCK GROUP
# 2024
age_df_2024 = api_lookup("B01002")

# name of block group and median age of person
age_by_block_group_2024 = age_df_2024[["NAME", "B01002_001E", "GEOID"]].sort_values(
    "B01002_001E"
)


age_by_block_group_2024.columns = ["name", "median_age_2024", "geoid"]

age_by_block_group_2024["median_age_2024"] = clean_attribute(
    age_by_block_group_2024["median_age_2024"]
)

# sanity check: looking at wampus:
age_by_block_group_2024[
    age_by_block_group_2024["name"].str.contains("(Block Group 2; Census Tract 6.0)")
]


# 2019
age_df_2019 = api_lookup("B01002", year=2019)

# name of block group and median age of person
age_by_block_group_2019 = age_df_2019[["NAME", "B01002_001E", "GEOID"]].sort_values(
    "B01002_001E"
)

age_by_block_group_2019.columns = ["name", "median_age_2019", "geoid"]

age_by_block_group_2019["median_age_2019"] = clean_attribute(
    age_by_block_group_2019["median_age_2019"]
)

# sanity check: looking at wampus:
age_by_block_group_2019[
    age_by_block_group_2019["name"].str.contains("(Block Group 2; Census Tract 6.0)")
]

# %%
# INCOME BY BLOCK GROUP
income_df_2024 = api_lookup("B19013")

income_by_block_group_2024 = income_df_2024[
    ["NAME", "B19013_001E", "GEOID"]
].sort_values("B19013_001E")

income_by_block_group_2024.columns = ["name", "median_household_income_2024", "geoid"]

income_by_block_group_2024["median_household_income_2024"] = clean_attribute(
    income_by_block_group_2024["median_household_income_2024"]
)

missing_wampus_rows = {
    "name": [
        "Block Group 2; Census Tract 6.03; Travis County; Texas",
        "Block Group 5; Census Tract 6.03; Travis County; Texas",
        "Block Group 3; Census Tract 6.04; Travis County; Texas",
        "Block Group 4; Census Tract 6.03; Travis County, Texas",
        "Block Group 1; Census Tract 11; Travis County, Texas",
    ],
    "median_household_income": [17656, 5287, 8761, 4823, 114115],
    "geoid": [],
}

# sanity check: looking at wampus:
income_by_block_group_2024[
    income_by_block_group_2024["name"].str.contains("Block Group 2; Census Tract 6.")
]

# 2019
income_df_2019 = api_lookup("B19013", year=2019)

income_by_block_group_2019 = income_df_2019[
    ["NAME", "B19013_001E", "GEOID"]
].sort_values("B19013_001E")

income_by_block_group_2019.columns = ["name", "median_household_income_2019", "geoid"]

income_by_block_group_2019["median_household_income_2019"] = clean_attribute(
    income_by_block_group_2019["median_household_income_2019"]
)

# sanity check: looking at wampus:
income_by_block_group_2019[
    income_by_block_group_2019["name"].str.contains("Block Group 2, Census Tract 6.")
]

# %%
# SCHOOL BY BLOCK GROUP
# 2024
school_df_2024 = api_lookup("B14007")

school_by_block_group_2024 = school_df_2024[
    ["NAME", "B14007_017E", "B14007_018E", "GEOID"]
].sort_values("B14007_017E", ascending=False)

school_by_block_group_2024.columns = [
    "name",
    "count_undergrad_2024",
    "count_grad_2024",
    "geoid",
]

school_by_block_group_2024["count_undergrad_2024"] = clean_attribute(
    school_by_block_group_2024["count_undergrad_2024"]
)

school_by_block_group_2024["count_grad_2024"] = clean_attribute(
    school_by_block_group_2024["count_grad_2024"]
)

# sanity check: looking at wampus:
school_by_block_group_2024[
    school_by_block_group_2024["name"].str.contains("Block Group 2; Census Tract 6.")
]

# 2019
school_df_2019 = api_lookup("B01002", year=2019)

# name of block group and count of people's education
school_by_block_group_2019 = school_df_2024[
    ["NAME", "B14007_017E", "B14007_018E", "GEOID"]
].sort_values("B14007_017E", ascending=False)

school_by_block_group_2019.columns = [
    "name",
    "count_undergrad_2019",
    "count_grad_2019",
    "geoid",
]

school_by_block_group_2019["count_undergrad_2019"] = clean_attribute(
    school_by_block_group_2019["count_undergrad_2019"]
)

school_by_block_group_2019["count_grad_2019"] = clean_attribute(
    school_by_block_group_2019["count_grad_2019"]
)

# sanity check: looking at wampus:
school_by_block_group_2019[
    school_by_block_group_2019["name"].str.contains("(Block Group 2; Census Tract 6.0)")
]

# %%
# combining to find all geoid
df = pd.concat(
    [
        age_by_block_group_2019[["geoid"]],
        age_by_block_group_2024[["geoid"]],
        income_by_block_group_2024[["geoid"]],
        income_by_block_group_2019[["geoid"]],
        school_by_block_group_2024[["geoid"]],
        school_by_block_group_2019[["geoid"]],
    ],
    ignore_index=True,
)

df = df.drop_duplicates().sort_values("geoid").reset_index(drop=True)

print(df.head())
print(len(df))

# %%
df = (
    df.merge(age_by_block_group_2024, on="geoid", how="left")
    .merge(income_by_block_group_2024, on="geoid", how="left")
    .merge(school_by_block_group_2024, on="geoid", how="left")
)

# %%
name_geoid_lookup = df[["name", "geoid"]]

# %%
age_by_block_group_2019.drop("name", inplace=True, axis=1)
income_by_block_group_2019.drop("name", inplace=True, axis=1)
school_by_block_group_2019.drop("name", inplace=True, axis=1)

df = (
    df.merge(age_by_block_group_2019, on="geoid", how="left")
    .merge(income_by_block_group_2019, on="geoid", how="left")
    .merge(school_by_block_group_2019, on="geoid", how="left")
)

df.head()

# %%
df["median_age_2024"] = df["median_age_2024"].fillna(df["median_age_2019"])
df["count_undergrad_2024"] = df["count_undergrad_2024"].fillna(
    df["count_undergrad_2019"]
)
df["count_grad_2024"] = df["count_grad_2024"].fillna(df["count_grad_2019"])

# inflation from 2019 to 2024
factor_2019_to_2024 = 462.5 / 375.8  # 1.23070782

df["median_household_income_2024"] = df["median_household_income_2024"].fillna(
    df["median_household_income_2019"] * factor_2019_to_2024
)

# %%
plt.hist(df["median_age_2024"])
plt.hist(df["count_undergrad_2024"])
plt.hist(df["count_grad_2024"])
plt.hist(df["median_household_income_2024"])

# %%
df = df[
    [
        "geoid",
        "median_age_2024",
        "median_household_income_2024",
        "count_undergrad_2024",
        "count_grad_2024",
    ]
]

df.columns = ["geoid", "age", "income", "undergrad", "grad"]

df = df.dropna()

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# %%
# your ACS dataframe
# should already look like:
# geoid | age | income | undergrad | grad

df["geoid"] = df["geoid"].astype(str)

# %%
# load block group shapefile
bg_shapes = gpd.read_file("../raw/block_shapes/tl_2024_48_bg.shp")

# %%
# inspect columns
print(bg_shapes.columns)
print(bg_shapes.head())

# %%
# make sure GEOID types match
bg_shapes["GEOID"] = bg_shapes["GEOID"].astype(str)

# %%
# optional: keep only Travis County
# Texas state FIPS = 48
# Travis County FIPS = 453
bg_shapes = bg_shapes[bg_shapes["COUNTYFP"] == "453"].copy()

# %%
# join your ACS data onto the shapes
map_df = bg_shapes.merge(
    df,
    left_on="GEOID",
    right_on="geoid",
    how="inner",  # use "left" if you want all shapes, even unmatched ones
)

# %%
print(map_df[["GEOID", "age", "income", "undergrad", "grad"]].head())
print(map_df.shape)

# %%
# %%
# %%
import folium

# project to lat/lon for folium
map_df = map_df.to_crs(epsg=4326)

# center on Travis County roughly
m = folium.Map(location=[30.3, -97.75], zoom_start=10)

folium.Choropleth(
    geo_data=map_df,
    data=map_df,
    columns=["GEOID", "income"],
    key_on="feature.properties.GEOID",
    fill_color="YlGnBu",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Median Household Income",
).add_to(m)

m
