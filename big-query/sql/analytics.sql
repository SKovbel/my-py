-- cumulative number of trips for each date/bike in 2015.
-- 22, 13:25:00, 2, 16, 2, 16
-- 25, 11:43:00, 77, 60, 77, 51
-- 25, 12:14:00, 60, 51, 77, 51
-- 29, 14:59:00, 46, 60, 46, 74
-- 29, 21:23:00, 60, 74, 46, 74
-- 32, 09:27:00, 36, 36, 36, 36
 SELECT bike_number,
        TIME(start_date) AS trip_time,
        start_station_id,
        end_station_id,
        FIRST_VALUE(start_station_id) OVER (
            PARTITION BY bike_number
            ORDER BY start_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS first_station_id,
        LAST_VALUE(end_station_id) OVER (
            PARTITION BY bike_number
            ORDER BY start_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ) AS last_station_id
   FROM bigquery-public-data.san_francisco.bikeshare_trips
  WHERE DATE(start_date) = '2015-10-25' 



-- Each row corresponds to calculate the cumulative number of trips for each date in 2015.
-- 2015-01-01, 181, 181
-- 2015-01-02, 428, 609
-- 2015-01-03, 283, 892
WITH trips_by_day AS (
    SELECT DATE(start_date) AS trip_date,
           COUNT(*) as num_trips
      FROM bigquery-public-data.san_francisco.bikeshare_trips
     WHERE EXTRACT(YEAR FROM start_date) = 2015
     GROUP BY trip_date
)
SELECT *,
       SUM(num_trips) OVER (
            ORDER BY trip_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ) AS cumulative_trips
  FROM trips_by_day


-- the average number of trips, for the preceding 3 days and the following 3 days
-- 2016-01-31, 36496.857142857145
-- 2016-01-04, 33296.285714285717
-- 2016-02-28, 42667.857142857145
WITH trips_by_day AS (
    SELECT DATE(trip_start_timestamp) AS trip_date,
           COUNT(*) as num_trips
      FROM bigquery-public-data.chicago_taxi_trips.taxi_trips
     WHERE trip_start_timestamp > '2016-01-01'
       AND trip_start_timestamp < '2016-04-01'
     GROUP BY trip_date
     ORDER BY trip_date
)
SELECT trip_date,
       AVG(num_trips) OVER (ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING) AS avg_num_trips
FROM trips_by_day


-- order of trips were taken from their respective community areas
-- 11.0, 2013-10-03 00:45:00+00:00, 2013-10-03 00:45:00+00:00, 1
-- 11.0, 2013-10-03 01:30:00+00:00, 2013-10-03 01:30:00+00:00, 2
-- 11.0, 2013-10-03 06:00:00+00:00, 2013-10-03 06:00:00+00:00, 3
-- 11.0, 2013-10-03 06:30:00+00:00, 2013-10-03 06:30:00+00:00, 4
-- 11.0, 2013-10-03 06:45:00+00:00, 2013-10-03 07:15:00+00:00, 5
SELECT pickup_community_area,
       trip_start_timestamp,
       trip_end_timestamp,
       RANK() OVER(PARTITION BY pickup_community_area ORDER BY trip_start_timestamp) AS trip_number
  FROM bigquery-public-data.chicago_taxi_trips.taxi_trips
 WHERE DATE(trip_start_timestamp) = '2013-10-03'


-- length of the break (in minutes) that the driver had before each trip started
-- 121, 2013-10-03 17:45:00 UTC, 2013-10-03 18:45:00 UTC, null
-- 121, 2013-10-03 20:30:00 UTC, 2013-10-03 20:45:00 UTC, 15	
-- 121, 2013-10-03 20:15:00 UTC, 2013-10-03 20:15:00 UTC, 30
-- 121, 2013-10-03 23:00:00 UTC, 2013-10-03 23:15:00 UTC, 0
SELECT taxi_id,
       trip_start_timestamp,
       trip_end_timestamp,
       TIMESTAMP_DIFF(
          trip_start_timestamp, 
          LAG(trip_end_timestamp, 1)  OVER (PARTITION BY taxi_id ORDER BY trip_start_timestamp), 
          MINUTE
       ) AS prev_break
   FROM bigquery-public-data.chicago_taxi_trips.taxi_trips
  WHERE DATE(trip_start_timestamp) = '2013-10-03' 