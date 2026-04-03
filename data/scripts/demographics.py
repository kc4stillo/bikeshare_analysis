# %%

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


# %%
# AGE BY BLOCK GROUP
# 2024
age_df_2024 = api_lookup("B01002")

# name of block group and median age of person
age_by_block_group_2024 = age_df_2024[["NAME", "B01002_001E", "GEOID"]].sort_values(
    "B01002_001E"
)

# sanity check: looking at wampus:
age_by_block_group_2024[
    age_by_block_group_2024["NAME"].str.contains("(Block Group 2; Census Tract 6.0)")
]

age_by_block_group_2024.columns = ["name", "median_age", "geoid"]

# 2019
age_df_2019 = api_lookup("B01002", year=2019)

# name of block group and median age of person
age_by_block_group_2019 = age_df_2019[["NAME", "B01002_001E", "GEOID"]].sort_values(
    "B01002_001E"
)

# sanity check: looking at wampus:
age_by_block_group_2019[
    age_by_block_group_2019["NAME"].str.contains("(Block Group 2; Census Tract 6.0)")
]

age_by_block_group_2019.columns = ["name", "median_age", "geoid"]


# %%
# INCOME BY BLOCK GROUP
income_df_2024 = api_lookup("B19013")

income_by_block_group_2024 = income_df_2024[
    ["NAME", "B19013_001E", "GEOID"]
].sort_values("B19013_001E")

# sanity check: looking at wampus:
income_by_block_group_2024[
    income_by_block_group_2024["NAME"].str.contains("Block Group 2; Census Tract 6.")
]

income_by_block_group_2024.columns = ["name", "median_household_income", "geoid"]

# 2019
income_df_2019 = api_lookup("B19013", year=2019)

income_by_block_group_2019 = income_df_2019[
    ["NAME", "B19013_001E", "GEOID"]
].sort_values("B19013_001E")

# sanity check: looking at wampus:
income_by_block_group_2019[
    income_by_block_group_2019["NAME"].str.contains("Block Group 2, Census Tract 6.")
]

income_by_block_group_2019.columns = ["name", "median_household_income", "geoid"]

# %%
# SCHOOL BY BLOCK GROUP
# 2024
school_df_2024 = api_lookup("B14007")

school_by_block_group_2024 = school_df_2024[
    ["NAME", "B14007_017E", "B14007_018E", "GEOID"]
].sort_values("B14007_017E", ascending=False)

# sanity check: looking at wampus:
school_by_block_group_2024[
    school_by_block_group_2024["NAME"].str.contains("Block Group 2; Census Tract 6.")
]

school_by_block_group_2024.columns = ["name", "count_undergrad", "count_grad", "geoid"]

# 2019
age_df_2019 = api_lookup("B01002", year=2019)

# name of block group and median age of person
age_by_block_group_2019 = age_df_2019[["NAME", "B01002_001E", "GEOID"]].sort_values(
    "B01002_001E"
)

# sanity check: looking at wampus:
age_by_block_group_2019[
    age_by_block_group_2019["NAME"].str.contains("(Block Group 2; Census Tract 6.0)")
]

age_by_block_group_2019.columns = ["name", "median_age", "geoid"]

# %%
