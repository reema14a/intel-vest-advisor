SELECT
  CASE
    WHEN predicted_risk_tolerance IN (1, 2, 3) THEN 'Conservative'
    WHEN predicted_risk_tolerance IN (4, 5, 6, 7) THEN 'Moderate'
    WHEN predicted_risk_tolerance IN (8, 9, 10) THEN 'Aggressive'
    ELSE 'Unknown'
  END AS risk_profile,
  predicted_risk_tolerance AS predicted_label,
  (
    SELECT prob, 
  FROM UNNEST(predicted_risk_tolerance_probs)
  WHERE label = predicted_risk_tolerance
  ) AS probability
FROM
  ML.PREDICT(MODEL
`{project_id}.{dataset_id}.{model_name}`,
(
    SELECT
  {age_group} AS age_group,
  {education_level} AS education_level,
  {income_group} AS income_bracket,
  {emergency_savings} AS emergency_savings,
  {retirement_planning} AS retirement_planning,
  {financial_literacy_score} AS financial_literacy_score
  
)
);
