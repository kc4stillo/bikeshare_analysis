# Cleaned Data Dictionary

## Overview

This dataset evaluates MetroBike station performance and site suitability using operational metrics and spatial accessibility scoring. Each row represents a single station.

---

## Identifiers and Metadata

**`id`**  
Type: Int64  
Unique station identifier. Rows with missing `id` values are removed during cleaning.

**`name`**  
Type: string  
Standardized station name. Cleaning rules applied:
- Lowercase
- `/` replaced with `_and_`
- `@` replaced with `_at_`
- Spaces replaced with `_`

**`active_date`**  
Type: datetime64  
Date the station became operational.

**`district`**  
Type: Int64  
City council district in which the station is located.

---

## Operational Performance Metrics

**`total_checkouts`**  
Type: float  
Total number of trips initiated at the station during the evaluation period.

**`total_docks`**  
Type: Int64  
Number of physical docks installed at the station.

**`trips_per_dock`**  
Type: float  
Calculated as `total_checkouts / total_docks`. Measures usage efficiency relative to capacity.

**`trips_per_dock_day`**  
Type: float  
Average daily trips per dock. Primary utilization metric.

---

## System Classification

**`ebs_station`**  
Type: int (0/1)  
Indicates whether the station is part of the existing bikeshare (EBS) footprint.  
1 = Yes, 0 = No.

---

## Performance Ranking

**`checkouts_rank_per_day`**  
Type: float  
Categorical score (1–3) based on checkout activity per day.  
Original thresholds:
- >5 = 3  
- 2–5 = 2  
- <1 = 1  

Threshold methodology is currently under review.

---

## Accessibility and Spatial Scoring Variables

Each access variable uses a 1–3 scoring scale:  
3 = High access (closest proximity)  
2 = Moderate access  
1 = Low access (furthest proximity)

**`transit_access_score`**  
Proximity to transit infrastructure.

**`jobs_access_score`**  
Proximity to major employment hubs.

**`households_access_score`**  
Proximity to residential population density.

**`low_income_access_score`**  
Proximity to low-income residential populations.

**`public_amenities_access_score`**  
Proximity to libraries, schools, recreation centers, and parks.

**`bike_infra_score`**  
Proximity to bikeable infrastructure and rider safety corridors.

**`retail_entertainment_access_score`**  
Proximity to retail and entertainment destinations.

**`existing_bikeshare_access_score`**  
Proximity to existing bikeshare stations; measures network connectivity.

---

## Composite Score

**`total_score`**  
Type: float  
Sum of categorical scoring variables. Represents overall station suitability under the current scoring framework.
