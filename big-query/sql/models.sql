-- https://cloud.google.com/bigquery/docs/create-machine-learning-model
CREATE MODEL `my.sample_model`
OPTIONS(model_type='logistic_reg') AS
SELECT
  IF(totals.transactions IS NULL, 0, 1) AS label,
  IFNULL(device.operatingSystem, "") AS os,
  device.isMobile AS is_mobile,
  IFNULL(geoNetwork.country, "") AS country,
  IFNULL(totals.pageviews, 0) AS pageviews
FROM
   `bigquery-public-data.google_analytics_sample.ga_sessions_20170801`
WHERE
  _TABLE_SUFFIX BETWEEN '20160801' AND '20170630'


-- evaluate the model   
SELECT
  *
FROM
  ML.EVALUATE(MODEL `my.sample_model`, (
SELECT
  IF(totals.transactions IS NULL, 0, 1) AS label,
  IFNULL(device.operatingSystem, "") AS os,
  device.isMobile AS is_mobile,
  IFNULL(geoNetwork.country, "") AS country,
  IFNULL(totals.pageviews, 0) AS pageviews
FROM
   `bigquery-public-data.google_analytics_sample.ga_sessions_20170801`
))

-- model to predict outcomes
SELECT
  country,
  SUM(predicted_label) as total_predicted_purchases
FROM
  ML.PREDICT(MODEL `my.sample_model`, (
SELECT
  IFNULL(device.operatingSystem, "") AS os,
  device.isMobile AS is_mobile,
  IFNULL(totals.pageviews, 0) AS pageviews,
  IFNULL(geoNetwork.country, "") AS country
FROM
   `bigquery-public-data.google_analytics_sample.ga_sessions_20170801`
))
GROUP BY country
ORDER BY total_predicted_purchases DESC
LIMIT 10


-- predict purchases per user
SELECT
  fullVisitorId,
  SUM(predicted_label) as total_predicted_purchases
FROM
  ML.PREDICT(MODEL `my.sample_model`, (
SELECT
  IFNULL(device.operatingSystem, "") AS os,
  device.isMobile AS is_mobile,
  IFNULL(totals.pageviews, 0) AS pageviews,
  IFNULL(geoNetwork.country, "") AS country,
  fullVisitorId
FROM
   `bigquery-public-data.google_analytics_sample.ga_sessions_20170801`
))
GROUP BY fullVisitorId
ORDER BY total_predicted_purchases DESC
LIMIT 10

