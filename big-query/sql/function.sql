SELECT
       DATE(start_date),
       TIME(start_date),
       EXTRACT(HOUR FROM trip_start_timestamp)
       EXTRACT(DATE FROM s.time_ts),
       EXTRACT(DAYOFWEEK FROM timestamp_of_crash),
       EXTRACT(MONTH FROM trip_start_timestamp)
       EXTRACT(YEAR FROM trip_start_timestamp)
       TIMESTAMP_DIFF(a.creation_date, q.creation_date, SECOND)
       TIMESTAMP_DIFF(a.creation_date, q.creation_date, MINUTE)

       -- group + window
       COUNT(*),
       MIN(d.creation_date),
       SUM(*),
       AVG(num_trips)

       -- window
       LAG(trip_end_timestamp, 1)
       RANK(),
       FIRST_VALUE(start_station_id)
       LAST_VALUE(start_station_id)
FROM `bigquery-public-data.hacker_news.comments`
WHERE u.creation_date >= '2019-01-01' and u.creation_date < '2019-02-01'
  AND EXTRACT(DATE FROM q.creation_date) = '2019-01-01'
GROUP BY parent
HAVING COUNT(id) > 10
ORDER BY num_accidents DESC
