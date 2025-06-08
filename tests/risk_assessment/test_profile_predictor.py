from agents.shared.model_utils import ModelTrainer
from utils.load_env import load_agent_env

import os


def test_predict_risk_profile():
    # Load .env specific to the agent
    load_agent_env(__file__, "risk_assessment")

    # Fetch from env
    project_id = os.getenv("GCP_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET")
    sql_base = os.getenv("SQL_BASE")

    assert project_id, "PROJECT_ID not loaded from .env"
    assert dataset_id, "DATASET_ID not loaded from .env"
    assert sql_base, "SQL_BASE_PATH not loaded from .env"

    trainer = ModelTrainer(
        project_id=project_id,
        dataset_id=dataset_id,
        sql_base=sql_base,
    )

    # Test with Sample input
    result = trainer.predict_risk_profile(
        model_name="boosted_tree_classifier",
        age=3,
        education=6,
        income=5,
        emergency_savings=1,
        retirement_planning=2,
        financial_literacy_score=2,
    )

    assert result is not None, "❌ Prediction returned None"
    assert "risk_profile" in result, "❌ Missing 'risk_profile' in result"
    assert "predicted_label" in result, "❌ Missing 'predicted_label' in result"
    assert "probability" in result, "❌ Missing 'probability' in result"

    print("✅ Predicted result:", result)
