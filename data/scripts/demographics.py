# %%
import pandas as pd
import requests

pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)


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
age_df_2024 = api_lookup("B01002")

# name of block group and median age of person
age_by_block_group_2024 = age_df_2024[["NAME", "B01002_001E"]].sort_values(
    "B01002_001E"
)

# sanity check: looking at wampus:
age_by_block_group_2024[
    age_by_block_group_2024["NAME"].str.contains("Block Group 2; Census Tract 6.0")
]


# %%
# INCOME BY BLOCK GROUP
income_df_2024 = api_lookup("B19013")

income_by_block_group_2024 = income_df_2024[["NAME", "B19013_001E"]].sort_values(
    "B19013_001E"
)

# sanity check: looking at wampus:
income_by_block_group_2024[
    income_by_block_group_2024["NAME"].str.contains("Block Group 2; Census Tract 6.")
]

# 2019
income_df_2019 = api_lookup("B19013", year=2019)

income_by_block_group_2019 = income_df_2019[["NAME", "B19013_001E"]].sort_values(
    "B19013_001E"
)

# sanity check: looking at wampus:
income_by_block_group_2019[
    income_by_block_group_2019["NAME"].str.contains("Block Group 2, Census Tract 6.")
]

income_by_block_group_2019.head()
# NAME	B19013_001E
# 17	Block Group 1, Census Tract 24.30, Travis Coun...	-666666666
# 16	Block Group 1, Census Tract 23.19, Travis Coun...	-666666666
# 15	Block Group 1, Census Tract 23.14, Travis Coun...	-666666666
# 14	Block Group 1, Census Tract 23.12, Travis Coun...	-666666666
# 13	Block Group 3, Census Tract 22.08, Travis Coun...	-666666666

# %%
# SCHOOL BY BLOCK GROUP
school_df_2024 = api_lookup("B14007")

school_by_block_group_2024 = school_df_2024[
    ["NAME", "B14007_017E", "B14007_018E"]
].sort_values("B14007_017E", ascending=False)

# sanity check: looking at wampus:
school_by_block_group_2024[
    school_by_block_group_2024["NAME"].str.contains("Block Group 2; Census Tract 6.")
]

school_by_block_group_2024.head()
# NAME	B14007_017E	B14007_018E
# 354	Block Group 2; Census Tract 24.45; Travis Coun...	98	24
# 568	Block Group 2; Census Tract 368; Travis County...	98	11
# 349	Block Group 3; Census Tract 24.43; Travis Coun...	98	0
# 618	Block Group 1; Census Tract 409; Travis County...	97	0
# 429	Block Group 1; Census Tract 315; Travis County...	96	12
