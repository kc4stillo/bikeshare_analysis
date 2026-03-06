import geopandas as gpd
import pandas as pd

bg = gpd.read_file(r"../raw/housing/block_shapes.shp")
bg.head()

df = pd.read_csv("../raw/housing/house_data.csv")


bg
# GISJOIN	STATEFP	COUNTYFP	TRACTCE	BLKGRPCE	GEOID	GEOIDFQ	NAMELSAD	MTFCC	FUNCSTAT	ALAND	AWATER	INTPTLAT	INTPTLON	Shape_Leng	Shape_Area	geometry
# 0	G48000109501001	48	001	950100	1	480019501001	1500000US480019501001	Block Group 1	G5030	S	15079442.0	134009.0	+32.0610630	-095.4957453	21507.629535	1.521346e+07	POLYGON ((44957.61 -610660.193, 44964.366 -610...
# 1	G48000109501002	48	001	950100	2	480019501002	1500000US480019501002	Block Group 2	G5030	S	228222245.0	5866624.0	+31.9532088	-095.4932299	91952.048893	2.340889e+08	POLYGON ((54310.155 -609957.262, 54311.071 -60...
# 2	G48000109501003	48	001	950100	3	480019501003	1500000US480019501003	Block Group 3	G5030	S	240004926.0	1863680.0	+31.9800959	-095.6230974	78877.041446	2.418686e+08	POLYGON ((44957.61 -610660.193, 44943.802 -610...

df
# GISJOIN	YEAR	STUSAB	REGIONA	DIVISIONA	STATE	STATEA	COUNTY	COUNTYA	COUSUBA	...	BTBGA	TL_GEO_ID	NAME_E	AUUDE001	AUUDE002	AUUDE003	NAME_M	AUUDM001	AUUDM002	AUUDM003
# 0	G48000109501001	2020-2024	TX	NaN	NaN	Texas	48	Anderson County	1	NaN	...	NaN	480019501001	Block Group 1, Census Tract 9501, Anderson Cou...	660	578	82	Block Group 1, Census Tract 9501, Anderson Cou...	171	160	51
# 1	G48000109501002	2020-2024	TX	NaN	NaN	Texas	48	Anderson County	1	NaN	...	NaN	480019501002	Block Group 2, Census Tract 9501, Anderson Cou...	989	898	91	Block Group 2, Census Tract 9501, Anderson Cou...	202	219	76
# 2	G48000109501003	2020-2024	TX	NaN	NaN	Texas	48	Anderson County	1	NaN	...	NaN	480019501003	Block Group 3, Census Tract 9501, Anderson Cou...	626	544	82	Block Group 3, Census Tract 9501, Anderson Cou...	174	167	52


housing = bg.merge(df[["GISJOIN", "AUUDE002"]], on="GISJOIN", how="left")
housing = housing[housing["COUNTYFP"] == "453"]

# %%
import folium
import numpy as np

# Clean / prep
housing = housing.copy()
housing["AUUDE002"] = pd.to_numeric(housing["AUUDE002"], errors="coerce").fillna(0)

# INTPTLAT/INTPTLON are strings like "+30.2178959"
housing["lat"] = (
    housing["INTPTLAT"].astype(str).str.replace("+", "", regex=False).astype(float)
)
housing["lon"] = (
    housing["INTPTLON"].astype(str).str.replace("+", "", regex=False).astype(float)
)

# Center map on Travis County block group points
center_lat = housing["lat"].mean()
center_lon = housing["lon"].mean()

m = folium.Map(
    location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron"
)

# Scale radii (sqrt makes huge values less overwhelming)
vals = housing["AUUDE002"].to_numpy()
vmax = vals.max() if vals.max() > 0 else 1

min_radius = 50  # meters
max_radius = 600  # meters


def scale_radius(x):
    # sqrt scaling -> then map to [min_radius, max_radius]
    return min_radius + (np.sqrt(x) / np.sqrt(vmax)) * (max_radius - min_radius)


for _, r in housing.iterrows():
    hh = float(r["AUUDE002"])
    folium.Circle(
        location=[r["lat"], r["lon"]],
        radius=scale_radius(hh),  # meters
        fill=True,
        fill_opacity=0.35,
        opacity=0.6,
        popup=f"{r.get('NAMELSAD', 'Block Group')}<br>Occupied units: {int(hh):,}",
    ).add_to(m)

m
