import pandas as pd

stations = pd.read_csv("../data/a_stations/stations.csv")
amenities = pd.read_csv("../data/b_amenities/clean/amenities.csv")
dining_halls = pd.read_csv("../data/b_amenities/clean/dining_halls.csv")
parks = pd.read_csv("../data/b_amenities/clean/dining_halls.csv")
demographics = pd.read_csv("../data/c_demographics/clean/demographics.csv")
ut_shape = pd.read_csv("../data/d_ut_shapes/clean/ut_shape.csv")
west_campus = pd.read_csv("../data/d_ut_shapes/clean/west_campus.csv")
jobs = pd.read_csv("../data/e_jobs/clean/jobs.csv")
retail = pd.read_csv("../data/f_retail/clean/retail.csv")
tranit = pd.read_csv("../data/g_transit/clean/transit.csv")

stations
