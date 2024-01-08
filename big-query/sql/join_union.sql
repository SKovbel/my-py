-- Usernames corresponding to users who wrote stories or comments on January 1, 2014.
-- We use UNION DISTINCT (instead of UNION ALL) to ensure that each user appears in the table at most once.
SELECT c.by
  FROM bigquery-public-data.hacker_news.comments AS c
 WHERE EXTRACT(DATE FROM c.time_ts) = '2014-01-01'
 UNION DISTINCT
SELECT s.by
  FROM bigquery-public-data.hacker_news.stories AS s
 WHERE EXTRACT(DATE FROM s.time_ts) = '2014-01-01'
