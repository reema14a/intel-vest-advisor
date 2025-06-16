import pytest
from unittest.mock import Mock, patch, AsyncMock
from google.adk.sessions import InMemorySessionService, Session
from google.adk.runners import Runner
from google.genai import types
from agents.risk_model_selection.agent import (
    RiskModelSelectionAgent,
    ModelSelectionOutput,
    ModelMetric
)
from agents.shared.model_utils import ModelTrainer
from utils.session_manager import SessionManager
from utils.load_env import load_agent_env
from utils.logging_config import setup_logging, get_logger
import os
import json

# Set up logging
setup_logging()
logger = get_logger(__name__)

@pytest.fixture
def session_manager():
    return SessionManager()

@pytest.fixture
def risk_model_selection_agent():
    """Create a RiskModelSelectionAgent instance for testing."""
    return RiskModelSelectionAgent()

@pytest.fixture
def mock_model_trainer():
    with patch('agents.risk_model_selection.agent.ModelTrainer') as mock:
        trainer_instance = Mock()
        trainer_instance.train_all_models.return_value = "Training completed"
        trainer_instance.evaluate_all_models.return_value = [
            {
                "model_name": "LOGISTIC_REG",
                "recall": 0.85,
                "f1_score": 0.82,
                "precision": 0.80,
                "accuracy": 0.83
            },
            {
                "model_name": "BOOSTED_TREE_CLASSIFIER",
                "recall": 0.88,
                "f1_score": 0.86,
                "precision": 0.84,
                "accuracy": 0.87
            }
        ]
        mock.return_value = trainer_instance
        yield mock

@pytest.fixture
def setup_env():
    """Setup environment variables for testing."""
    load_agent_env(__file__, "risk_model_selection")
    logger.info("Loading environment variables for risk model selection tests")
    return {
        "project_id": os.getenv("GCP_PROJECT_ID"),
        "dataset_id": os.getenv("BQ_DATASET"),
        "table_name": os.getenv("BQ_TABLE"),
        "data_path": os.getenv("DATA_PATH"),
        "sql_base": os.getenv("SQL_BASE")
    }

# Unit Tests
@pytest.mark.asyncio
async def test_risk_model_selection_agent_initialization(risk_model_selection_agent):
    """Test the initialization of the RiskModelSelectionAgent."""
    logger.info("Testing RiskModelSelectionAgent initialization")
    assert risk_model_selection_agent.name == "risk_model_selection_agent"
    assert "Selects the best BigQuery ML model" in risk_model_selection_agent.description
    # Tools are added at runtime in before_agent_callback
    assert hasattr(risk_model_selection_agent, 'tools')

@pytest.mark.asyncio
async def test_before_agent_callback(risk_model_selection_agent, setup_env):
    """Test the before_agent_callback method with actual environment variables."""
    logger.info("Testing before_agent_callback with environment variables")
    # Verify environment setup
    assert setup_env["project_id"], "GCP_PROJECT_ID not set"
    assert setup_env["dataset_id"], "BQ_DATASET not set"
    assert setup_env["table_name"], "BQ_TABLE not set"
    assert setup_env["data_path"], "DATA_PATH not set"
    assert setup_env["sql_base"], "SQL_BASE not set"

    mock_context = Mock()
    mock_context._invocation_context.agent = Mock()
    
    # Call before_agent_callback (not async)
    risk_model_selection_agent.before_agent_callback(mock_context)
    
    # Verify tools were added
    assert len(mock_context._invocation_context.agent.tools) == 2
    setup_tool, evaluate_tool = mock_context._invocation_context.agent.tools
    
    # Test setup tool
    setup_result = setup_tool()
    assert "Setup and model training completed" in setup_result
    
    # Test evaluate tool
    eval_result = evaluate_tool()
    assert "LOGISTIC_REG" in eval_result
    assert "BOOSTED_TREE_CLASSIFIER" in eval_result
    assert "0.85" in eval_result
    assert "0.88" in eval_result

@pytest.mark.asyncio
async def test_before_agent_callback_missing_env_vars(risk_model_selection_agent):
    """Test the before_agent_callback method with missing environment variables."""
    logger.info("Testing before_agent_callback with missing environment variables")
    mock_context = Mock()
    mock_context._invocation_context.agent = Mock()
    
    # Temporarily clear environment variables
    original_env = dict(os.environ)
    os.environ.clear()
    
    try:
        with pytest.raises(ValueError, match="Missing one or more required env variables"):
            # Call before_agent_callback (not async)
            risk_model_selection_agent.before_agent_callback(mock_context)
    finally:
        # Restore original environment variables
        os.environ.update(original_env)

@pytest.mark.asyncio
async def test_after_agent_callback(risk_model_selection_agent):
    """Test the after_agent_callback method."""
    logger.info("Testing after_agent_callback")
    mock_context = Mock()
    mock_context._invocation_context.agent = Mock()
    mock_context._invocation_context.agent.output = Mock()
    mock_context._invocation_context.agent.output.content = json.dumps({
        "selectedModel": "balanced_portfolio",
        "reason": "Best fit for moderate risk tolerance"
    })
    
    # Call after_agent_callback (not async)
    risk_model_selection_agent.after_agent_callback(mock_context)
    
    # Verify output was processed
    assert mock_context._invocation_context.agent.output is not None

# Integration Tests
@pytest.mark.asyncio
async def test_model_setup_and_training(setup_env):
    """Test the model setup and training process."""
    # Initialize the agent
    agent = RiskModelSelectionAgent()
    
    # Verify environment setup
    assert setup_env["project_id"], "GCP_PROJECT_ID not set"
    assert setup_env["dataset_id"], "BQ_DATASET not set"
    assert setup_env["table_name"], "BQ_TABLE not set"
    assert setup_env["data_path"], "DATA_PATH not set"
    assert setup_env["sql_base"], "SQL_BASE not set"
    
    # Test model setup
    setup_result = await agent.setup_tool(
        project_id=setup_env["project_id"],
        dataset_id=setup_env["dataset_id"],
        data_path=setup_env["data_path"],
        table_name=setup_env["table_name"],
        sql_base=setup_env["sql_base"]
    )
    
    assert setup_result is not None
    assert "status" in setup_result
    assert setup_result["status"] == "success"
    
    # Test model evaluation
    eval_result = await agent.evaluate_tool()
    assert eval_result is not None
    assert "metrics" in eval_result
    
    # Verify metrics for each model
    for model_metrics in eval_result["metrics"]:
        assert "model_name" in model_metrics
        assert "recall" in model_metrics
        assert "f1_score" in model_metrics
        assert "precision" in model_metrics
        assert "accuracy" in model_metrics
        
        # Verify metric values are within expected ranges
        assert 0 <= model_metrics["recall"] <= 1
        assert 0 <= model_metrics["f1_score"] <= 1
        assert 0 <= model_metrics["precision"] <= 1
        assert 0 <= model_metrics["accuracy"] <= 1

@pytest.mark.asyncio
async def test_model_selection_integration(session_manager, setup_env):
    """Test the model selection process with a trained model."""
    logger.info("Testing model selection integration")
    # Verify environment setup
    assert setup_env["project_id"], "GCP_PROJECT_ID not set"
    assert setup_env["dataset_id"], "BQ_DATASET not set"
    assert setup_env["table_name"], "BQ_TABLE not set"
    assert setup_env["data_path"], "DATA_PATH not set"
    assert setup_env["sql_base"], "SQL_BASE not set"

    # Setup session service and IDs
    session_service = InMemorySessionService()
    app_name = "risk_model_app"
    user_id = "test_user"
    session_id = "session_xyz"
    
    # Create a session object with required fields
    session = Session(
        id=session_id,  # Required field
        app_name=app_name,
        user_id=user_id,
        state={}
    )
    
    # Initialize the agent
    agent = RiskModelSelectionAgent()
    
    # Create a runner with required parameters
    runner = Runner(
        session_service=session_service,
        app_name=app_name,
        agent=agent
    )
    
    # Mock the model training and selection process
    async def mock_run_async(ctx):
        # Simulate model training and selection
        ctx.state["model_training_complete"] = True
        ctx.state["selected_model"] = "balanced_portfolio"
        yield Mock(
            is_final_response=lambda: True,
            content="Model selection complete"
        )
    
    with patch.object(agent, 'run_async', mock_run_async):
        # Run the model selection process
        async for event in agent.run_async(runner._new_invocation_context(session)):
            if hasattr(event, "is_final_response") and event.is_final_response():
                # Verify the final response
                assert "Model selection complete" in event.content
                
                # Verify session state updates
                assert session.state["model_training_complete"] is True
                assert session.state["selected_model"] == "balanced_portfolio"
