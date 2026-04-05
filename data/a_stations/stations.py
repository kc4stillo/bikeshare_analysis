# %%
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)


# %%
def clean_station_name(s):
    if pd.isna(s):
        return s
    s = str(s).strip()
    s = s.replace("\t", "")
    s = " ".join(s.split())  # remove weird extra spaces

    s = s.lower()
    s = s.replace("/", "_")
    s = s.replace("@", "_at_")
    s = s.replace(" ", "_")
    s = s.replace(".", "")
    s = s.replace("  ", " ")
    s = s.replace("__", "_")

    return s


# %%
scores = pd.read_excel("raw/curr_station_rubric.xlsx", header=2)
# only current stations
scores = scores.iloc[:72].copy()

trips = pd.read_csv("raw/tips_per_station.csv")

coords = pd.read_excel("raw/curr_station_coords.xlsx", header=1)
coords.columns = ["name", "lat", "lon"]

#  clean names first
scores["name"] = scores["name"].apply(clean_station_name)
trips["name"] = trips["name"].apply(clean_station_name)
coords["name"] = coords["name"].apply(clean_station_name)

# drop rows from scores that are planned
scores = scores.drop(scores[scores["name"] == "barton_springs_azie_morton"].index)
scores = scores.drop(scores[scores["name"] == "e_5th_neches_at_downtown_station"].index)
scores = scores.drop(scores[scores["name"] == "e_8th_trinity"].index)


# drop rows from trips and coords based on notes
drop_names = [
    "warehouse_sation",
    "test-lucj",
    "e_7th_congress",
    "e_6th_robert_t_martinez",
    "e_7th_congress",
    "e_7th_pleasant_valley",
    "webberville_northwestern",
    "lakeshore_austin_hostel",
    "e_cesar_chavez_pleasant_valley",
    "e_8th_san_jacinto",
    "e_5th_shady_and_eastside_bus_plaza",
    "30th_whitis",
    "w_4th_guadalupe_and_republic_square",
]

# TODO: FIX THIS AND MAKE IT LESS UGLY
manual_coords = pd.DataFrame(
    {
        "name": [
            "e_10th_red_river",
            "w_11th_congress_at_the_texas_capitol",
            "e_11th_salina",
            "e_11th_san_jacinto",
            "e_12th_san_jacinto_at_state_cap_visitors_garage",
            "e_13th_trinity_at_waterloo_greenway",
            "w_21st_guadalupe",
            "e_21st_speedway_at_pcl",
            "w_21st_university",
            "w_225_rio_grande",
            "w_22nd_pearl",
            "w_23rd_pearl",
            "e_23rd_san_jacinto_at_dkr_stadium",
            "w_26th_nueces",
            "w_28th_rio_grande",
            "e_2nd_pedernales",
            "e_2nd_congress",
            "w_2nd_lavaca_at_city_hall",
            "w_3rd_nueces",
            "w_3rd_west",
            "w_4th_congress",
            "e_4th_sabine",
            "w_5th_bowie",
            "w_5th_campbell",
            "e_6th_chalmers",
            "w_6th_congress",
            "w_6th_lavaca",
            "e_6th_trinity",
            "w_6th_west",
            "w_8th_congress",
            "e_8th_lavaca",
            "e_8th_red_river",
            "e_8th_san_jacinto",
            "w_9th_henderson",
            "barton_springs_pool",
            "barton_springs_bouldin_at_palmer_auditorium",
            "barton_springs_kinney",
            "barton_springs_sterzing",
            "cesar_chavez_congress",
            "dean_keeton_park_place",
            "dean_keeton_robert_dedman_dr",
            "dean_keeton_speedway",
            "dean_keeton_whitis",
            "e_11th_san_marcos",
            "e_11th_victory_grill",
            "e_4th_chicon",
            "e_5th_broadway",
            "e_6th_medina",
            "e_6th_pedernales",
            "e_6th_robert_t_martinez",
            "electric_drive_at_pfluger_ped_bridge",
            "guadalupe_west_mall_at_university_co_op",
            "hollow_creek_barton_hills",
            "lake_austin_blvd_deep_eddy",
            "lakeshore_austin_hostel",
            "lakeshore_pleasant_valley",
            "nash_hernandez_east_at_rbj_south",
            "one_texas_center",
            "plaza_saltillo",
            "rainey_cummings",
            "rainey_driskill",
            "riverside_south_lamar",
            "rosewood_angelina",
            "s_1st_riverside_at_long_center",
            "south_congress_at_bouldin_creek",
            "south_congress_academy",
            "south_congress_barton_springs",
            "south_congress_elizabeth",
            "south_congress_james",
            "south_congress_mary",
            "veterans_atlanta_at_mopac_ped_bridge",
            "zilker_park",
            "w_30th_whitis",
            "e_5th_shady",
            "webberville_northwestern",
            "webberville_neal",
            "republic_square",
            "e_cesar_chavez_pleasant_valley",
            "e_cesar_chavez_e_7th",
            "e_11th_waller",
            "e_6th_chicon",
            "lakeshore_lady_bird_ln",
            "w_16th_san_antonio",
            "w_23rd_san_gabriel",
            "w_7th_congress_(w_6th_congress)",
        ],
        "lat": [
            30.270155,
            30.272551,
            30.266422,
            30.271838,
            30.273499,
            30.274063,
            30.283985,
            30.282894,
            30.283535,
            30.286170,
            30.285384,
            30.287401,
            30.285500,
            30.290682,
            30.293155,
            30.255424,
            30.264095,
            30.264760,
            30.266972,
            30.267756,
            30.266343,
            30.2650342,
            30.269210,
            30.274782,
            30.262684,
            30.268257,
            30.268949,
            30.267190,
            30.270466,
            30.269793,
            30.270522,
            30.268550,
            30.268800,
            30.272176,
            30.264500,
            30.259660,
            30.261928,
            30.264401,
            30.263344,
            30.289310,
            30.287850,
            30.289510,
            30.289822,
            30.269669,
            30.268959,
            30.259718,
            30.256297,
            30.264582,
            30.258999,
            30.261165,
            30.267064,
            30.285664,
            30.261617,
            30.278107,
            30.244687,
            30.242667,
            30.252056,
            30.257653,
            30.262116,
            30.255942,
            30.260810,
            30.264351,
            30.269014,
            30.259384,
            30.255021,
            30.252232,
            30.258694,
            30.248923,
            30.251048,
            30.244929,
            30.274475,
            30.265882,
            30.295427,
            30.252123,
            30.263061,
            30.267506,
            30.267416,
            30.252951,
            30.260104,
            30.26899897139031,
            30.26125588077636,
            30.244727662839196,
            30.27941719277451,
            30.287119781194694,
            30.268310763977247,
        ],
        "lon": [
            -97.735364,
            -97.741225,
            -97.721623,
            -97.738019,
            -97.738097,
            -97.736657,
            -97.741980,
            -97.737349,
            -97.739530,
            -97.745201,
            -97.746661,
            -97.747718,
            -97.733492,
            -97.742902,
            -97.744154,
            -97.716667,
            -97.743535,
            -97.746728,
            -97.749462,
            -97.751763,
            -97.743817,
            -97.7391681,
            -97.753495,
            -97.764786,
            -97.724284,
            -97.742859,
            -97.745223,
            -97.739245,
            -97.750461,
            -97.742239,
            -97.744618,
            -97.736492,
            -97.739810,
            -97.752456,
            -97.771359,
            -97.753445,
            -97.761131,
            -97.764315,
            -97.745094,
            -97.733037,
            -97.728541,
            -97.736535,
            -97.740468,
            -97.730623,
            -97.728434,
            -97.723198,
            -97.710115,
            -97.731650,
            -97.714794,
            -97.721000,
            -97.754820,
            -97.741792,
            -97.772596,
            -97.772695,
            -97.723109,
            -97.717651,
            -97.734532,
            -97.748980,
            -97.727410,
            -97.739899,
            -97.738075,
            -97.756216,
            -97.724294,
            -97.749726,
            -97.747579,
            -97.748805,
            -97.746220,
            -97.750206,
            -97.749253,
            -97.751334,
            -97.769892,
            -97.768312,
            -97.739347,
            -97.698053,
            -97.713433,
            -97.707997,
            -97.747417,
            -97.712467,
            -97.709724,
            -97.72842914529065,
            -97.72132623864785,
            -97.72285895003382,
            -97.74371821617328,
            -97.74793884604874,
            -97.74287793292342,
        ],
    }
)

trips = trips[~trips["name"].isin(drop_names)]
coords = coords[~coords["name"].isin(drop_names)]

coords = (pd.concat([coords, manual_coords])).drop_duplicates(subset=["name"])

# manual fixes for mismatched station names
# key = name in trips/coords
# value = matching name in scores

name_map = {
    "w_6th_congress": "w_7th_congress_(w_6th_congress)",
    "e_12th_san_jacinto_at_state_capitol_visitors_g": "e_12th_san_jacinto_at_state_cap_visitors_garage",
}

trips["name"] = trips["name"].replace(name_map)


#  merge
df = scores.merge(trips, on="name", how="left")
df = df.merge(coords, on="name", how="left")

df = df.drop_duplicates(subset=["name"])

df[df["trips"].isna()]
df[df["lat"].isna()]

df.columns

# %%
df.drop(
    [
        "id",
        "Active Date",
        "total Checkouts",
        "trips per dock",
        "trips per dock/day",
        "Checkouts Rankings; per day >5=3; 2-5=2; <1=1 ",
        "Co-locate to Transit (at transit =3; <1/4 mi = 2; >1/4 mi = 1)",
        "Access to Jobs (Major employment hubs)  (1/4 mi = 3; 1/2 mi = 2; >1/2 = 1)",
        "Access to Households  (1/4 mi = 3; 1/2 mi = 2;  >1/2 = 1)",
        "Access to low income residents (1/4 mi = 3; 1/2 mi = 2; >1/2 = 1)",
        "Access to Public amenities (libraries, schools, Rec Centers, parks)  (1/4 mi = 3; 1/2 mi = 2; >1/2 = 1)",
        "Access to retail or entertainment  (1/4 mi = 3; 1/2 mi = 2; >1/2 = 1)",
        "Access to existing Bikeshare footprint - 1/4 mi = 3; 1/2 mi = 2; >1/2 = 1",
        "Total Score",
        "EBS STATION",
    ],
    inplace=True,
    axis=1,
)

df["trips_per_dock"] = df["trips"] / df["total Docks"]

df.columns

df = df.rename(
    columns={
        "Districts": "district",
        "total Docks": "docks",
        "Bikeable infrastructure (rider saftey)  (1/4 mi = 3; 1/2 mi = 2; >1/2 = 1)": "bikeable_infrastructure",
    }
)

df.head()

df.to_csv("stations.csv")
