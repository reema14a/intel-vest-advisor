from agents.shared.model_utils import ModelTrainer
from utils.load_env import load_agent_env

import os


def test_train_and_evaluate_models():
    # Load .env specific to the agent
    load_agent_env(__file__, "risk_model_selection")

    # Fetch from env
    project_id = os.getenv("GCP_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET")
    table_name = os.getenv("BQ_TABLE")
    data_path = os.getenv("DATA_PATH")
    sql_base = os.getenv("SQL_BASE")

    # assert project_id, "PROJECT_ID not loaded from .env"
    # assert dataset_id, "DATASET_ID not loaded from .env"
    # assert table_name, "TABLE_NAME not loaded from .env"
    # assert data_path, "DATA_PATH not loaded from .env"
    # assert sql_base, "SQL_BASE_PATH not loaded from .env"

    trainer = ModelTrainer(
        project_id=project_id,
        dataset_id=dataset_id,
        data_path=data_path,
        table_name=table_name,
        sql_base=sql_base,
    )

    trainer.train_all_models()
    results = trainer.evaluate_all_models()

    assert len(results) == 3

    for row in results:
        assert "roc_auc" in dict(row)
        assert "accuracy" in dict(row)
