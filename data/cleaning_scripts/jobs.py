import pandas as pd

TRAVIS_FIPS = 48453  # Travis County, TX

df1 = pd.read_csv("../raw/jobs/blocks.csv", dtype={"tabblk2020": str})
df2 = pd.read_csv("../raw/jobs/jobs.csv", dtype={"w_geocode": str})

# Keep only Travis County blocks (crosswalk has county)
xwalk_travis = df1.loc[
    df1["cty"] == TRAVIS_FIPS, ["tabblk2020", "blklatdd", "blklondd"]
].copy()

# WAC (jobs) -> rename join key to match
wac = df2[["w_geocode", "C000"]].rename(columns={"w_geocode": "tabblk2020"})

# Merge: only jobs that fall on Travis blocks will match
jobs_blocks_travis = wac.merge(xwalk_travis, on="tabblk2020", how="inner")

# Quick checks
print("Travis blocks in xwalk:", len(xwalk_travis))
print("Jobs blocks matched in Travis:", len(jobs_blocks_travis))
print(jobs_blocks_travis.head())

jobs_blocks_travis = jobs_blocks_travis[["C000", "blklatdd", "blklondd"]]
jobs_blocks_travis.columns = ["job_count", "lat", "lon"]

jobs_blocks_travis.to_csv("../cleaned/jobs.csv", index=False)
