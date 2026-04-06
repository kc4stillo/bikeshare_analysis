# JOBS
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd().parents[2]))

from utilities import count_within_radius

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# %%
stations = pd.read_csv("../../d_ut_shapes/clean/stations.csv")
jobs = pd.read_csv("../clean/jobs.csv")

stations.head()
# name	docks	bikeable_infrastructure	trips	lat	lon	trips_per_dock	nearest_dining_hall_m	nearest_amenity_m	nearest_park_m	avg_dist_3_amenities_m	park_area_within_275m	park_area_within_550m	age	income	population	undergrad	grad	undergrad_percentage	grad_percentage
# 0	barton_springs_pool	11	2.0	3613	30.264500	-97.771359	328.454545	3903.249201	48.268192	0.000000	142.871452	234168.308687	743414.143292	34.4	62031.0	869.0	25.0	25.0	0.028769	0.028769
# 1	barton_springs_bouldin_at_palmer_auditorium	15	3.0	5037	30.259660	-97.753445	335.800000	3047.383588	201.123665	4.855305	219.207081	114251.699654	332825.029494	35.8	124453.0	1245.0	7.0	26.0	0.005622	0.020884
# 2	barton_springs_kinney	11	3.0	2774	30.261928	-97.761131	252.181818	3307.996951	99.765738	271.389154	113.984834	194.525128	216223.521587	35.3	148333.0	1469.0	0.0	51.0	0.000000	0.034717
# 3	cesar_chavez_congress	11	3.0	3695	30.263344	-97.745094	335.909091	2325.915667	90.330300	0.000000	94.446839	21829.559364	105804.907720	37.0	155363.0	758.0	0.0	31.0	0.000000	0.040897
# 4	dean_keeton_park_place	15	3.0	4333	30.289310	-97.733037	288.866667	661.140420	103.696535	17.800356	142.264923	30366.151262	38498.217357	23.7	59138.0	865.0	421.0	98.0	0.486705	0.113295


jobs.head()
# 	job_count	lat	lon
# 0	40	30.334208	-97.755003
# 1	55	30.336065	-97.755197
# 2	4	30.326063	-97.747348
# 3	99	30.326369	-97.749075
# 4	11	30.321450	-97.748465


stations = count_within_radius(
    stations, jobs, radius_m=275, output_col="jobs_count_within_275m"
)
stations = count_within_radius(
    stations, jobs, radius_m=550, output_col="jobs_count_within_550m"
)

# %%
stations.to_csv("../clean/stations.csv", index=False)
