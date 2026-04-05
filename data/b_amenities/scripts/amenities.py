import osmnx as ox

place = "Travis County, Texas, USA"
tags = {
    "amenity": [
        "library",
        "community_centre",
        "social_centre",
        "arts_centre",
        "swimming_pool",
        "public_bath",
        "gym",
        "fitness_centre",
        "sports_centre",
        "recreation_ground",
        "youth_centre",
        "senior_centre",
        "event_venue",
        "townhall",
        "courthouse",
        "post_office",
        "fire_station",
        "police",
        "clinic",
        "hospital",
        "pharmacy",
    ],
    "leisure": [
        "swimming_pool",
        "sports_centre",
        "fitness_centre",
        "pitch",
        "track",
    ],
}

pois = ox.features_from_place(place, tags)

keep = ["geometry", "name", "amenity", "leisure"]
pois_small = pois[[c for c in keep if c in pois.columns]].copy()

pois = pois_small[
    pois_small["amenity"].isin(
        [
            "library",
            "community_centre",
            "social_centre",
            "arts_centre",
            "swimming_pool",
            "public_bath",
            "gym",
            "fitness_centre",
            "sports_centre",
            "recreation_ground",
            "youth_centre",
            "senior_centre",
            "event_venue",
            "townhall",
            "courthouse",
            "post_office",
            "fire_station",
            "police",
            "clinic",
            "hospital",
            "pharmacy",
        ]
    )
    | pois_small["leisure"].isin(
        [
            "swimming_pool",
            "sports_centre",
            "fitness_centre",
            "pitch",
            "track",
        ]
    )
].copy()

pois["geometry"] = pois.geometry.representative_point()
pois = pois.drop_duplicates(subset=["geometry"])

pois["type"] = pois["amenity"].fillna(pois["leisure"])

pois = pois.drop(columns=["amenity", "leisure"]).reset_index(drop=True)

pois["lon"] = pois.geometry.x
pois["lat"] = pois.geometry.y

amenities = pois[["name", "lat", "lon", "type"]]

amenities["name"] = amenities["name"].fillna(amenities["type"])

amenities.to_csv("../clean/amenities.csv", index=False)
