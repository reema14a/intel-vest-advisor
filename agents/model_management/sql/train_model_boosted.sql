CREATE OR REPLACE MODEL `{project_id}.{dataset_id}.model_boosted_tree_classifier`
OPTIONS
(
  model_type = 'boosted_tree_classifier',
  input_label_cols = ['risk_tolerance'],
  max_iterations = 10,
  early_stop = TRUE,
  data_split_method = 'NO_SPLIT',
  enable_global_explain = FALSE
) AS
SELECT *
FROM `{project_id}.{dataset_id}.{table_name}`;