import pytest
import os
from utils.load_env import load_agent_env
from utils.logging_config import setup_logging
from agents.shared.model_utils import ModelTrainer
import logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

def test_train_and_evaluate_models():
    # Load environment variables
    env = load_agent_env(__file__, "risk_model_selection", env="dev")
    logger.info(f"Running test in {env} environment")
    
    # Fetch from env
    project_id = os.getenv("GCP_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET")
    table_name = os.getenv("BQ_TABLE")
    data_path = os.getenv("DATA_PATH")
    sql_base = os.getenv("SQL_BASE")
    
    logger.info(f"Starting model training with project_id: {project_id}")
    logger.debug(f"Configuration: dataset={dataset_id}, table={table_name}, data_path={data_path}")
    
    # Validate environment variables
    assert project_id, "GCP_PROJECT_ID not loaded from .env"
    assert dataset_id, "BQ_DATASET not loaded from .env"
    assert table_name, "BQ_TABLE not loaded from .env"
    assert data_path, "DATA_PATH not loaded from .env"
    assert sql_base, "SQL_BASE not loaded from .env"
    
    trainer = ModelTrainer(
        project_id=project_id,
        dataset_id=dataset_id,
        data_path=data_path,
        table_name=table_name,
        sql_base=sql_base,
    )
    
    try:
        trainer.train_all_models()
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}", exc_info=True)
        raise

    results = trainer.evaluate_all_models()

    assert len(results) == 3

    for row in results:
        assert "roc_auc" in dict(row)
        assert "accuracy" in dict(row)
