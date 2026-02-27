import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

file_path = "../raw/scoring/raw_scores_with_coords.csv"
df = pd.read_csv(file_path)

# %%
# drop checkout rankings
df = df.drop("Checkouts Rankings; per day >5=3; 2-5=2; <1=1 ", axis=1)

# %%
