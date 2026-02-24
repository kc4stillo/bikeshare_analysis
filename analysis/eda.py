# %%
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../data/cleaned/current_stations_cleaned.csv")
project_stations = pd.read_csv("../data/cleaned/projected_stations_cleaned.csv")

df.dtypes

# %%
counts = df["district"].value_counts().sort_index()

plt.figure()
plt.bar(counts.index.astype(str), counts.values)
plt.xlabel("District")
plt.ylabel("# Stations")
plt.title("Stations per District")
plt.show()

# %%
plt.figure()
plt.hist(df["total_checkouts"], bins=20)
plt.xlabel("Total Checkouts")
plt.ylabel("Count of Stations")
plt.title("Distribution of Total Checkouts")
plt.show()

# %%
plt.figure()
plt.hist(df["trips_per_dock_day"], bins=20)
plt.xlabel("Trips per Dock per Day")
plt.ylabel("Count of Stations")
plt.title("Distribution of Trips per Dock per Day")
plt.show()


#  %%
plt.figure()
plt.boxplot(df["trips_per_dock_day"].dropna(), vert=True)
plt.ylabel("Trips per Dock per Day")
plt.title("Outliers in Trips per Dock per Day")
plt.show()

# %%
score_cols = [
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
key_cols = [
    "total_checkouts",
    "total_docks",
    "trips_per_dock",
    "trips_per_dock_day",
] + score_cols

corr = df[key_cols].corr(numeric_only=True)

plt.figure(figsize=(9, 7))
plt.imshow(corr, aspect="auto")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.index)), corr.index)
plt.title("Correlation Matrix (Usage + Scores)")
plt.colorbar()
plt.tight_layout()
plt.show()
# %%
