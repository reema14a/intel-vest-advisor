CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.model_{model_type}`
OPTIONS
(
  model_type='{model_type}',
  input_label_cols=['risk_tolerance']
) AS
SELECT *
FROM `{project_id}.{dataset_id}.{table_name}`;