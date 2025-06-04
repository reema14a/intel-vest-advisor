SELECT
    SUBSTR('{model_name}', 7) AS model_name,
    roc_auc,
    log_loss,
    precision,
    recall,
    accuracy,
    f1_score
FROM
    ML.EVALUATE(MODEL
`{project_id}.{dataset_id}.{model_name}`)
ORDER BY
  roc_auc DESC