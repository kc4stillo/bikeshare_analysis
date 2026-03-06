import numpy as np
import osmnx as ox

place = "Travis County, Texas, USA"
tags = {
    "shop": True,
    "amenity": [
        "restaurant",
        "cafe",
        "bar",
        "pub",
        "fast_food",
        "ice_cream",
        "cinema",
        "theatre",
        "nightclub",
        "brewery",
    ],
    "tourism": ["museum", "gallery", "attraction"],
}

pois = ox.features_from_place(place, tags)

keep = ["geometry", "name", "shop", "amenity", "tourism"]
pois_small = pois[[c for c in keep if c in pois.columns]].copy()

pois = pois_small[
    pois_small["shop"].notna()
    | pois_small["amenity"].isin(
        [
            "restaurant",
            "cafe",
            "bar",
            "pub",
            "fast_food",
            "ice_cream",
            "cinema",
            "theatre",
            "nightclub",
            "brewery",
        ]
    )
    | pois_small["tourism"].isin(["museum", "gallery", "attraction"])
].copy()

pois["geometry"] = pois.geometry.representative_point()
pois = pois.drop_duplicates(subset=["geometry"])


conds = [
    pois["shop"].notna(),
    pois["amenity"].notna(),
    pois["tourism"].notna(),
]
choices = [
    "shop_" + pois["shop"].astype(str),
    "amenity_" + pois["amenity"].astype(str),
    "tourism_" + pois["tourism"].astype(str),
]

pois["type"] = np.select(conds, choices, default=np.nan)

# (optional) drop the originals
pois = pois.drop(columns=["shop", "amenity", "tourism"]).reset_index(drop=True)

pois = pois.copy()

pois["lon"] = pois.geometry.x
pois["lat"] = pois.geometry.y

retail = pois[["name", "lat", "lon", "type"]]

retail.to_csv("../cleaned/retail/retail.csv", index=False)
