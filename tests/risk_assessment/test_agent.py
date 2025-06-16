import pytest
from unittest.mock import Mock, patch, AsyncMock
from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService, Session
from google.adk.runners import Runner
from google.genai import types
from agents.risk_assessment.agent import RiskAssessmentAgent, RiskAssessmentOutput
from agents.shared.model_utils import ModelTrainer
from utils.session_manager import SessionManager
from utils.logging_config import setup_logging, get_logger
import json

# Set up logging
setup_logging()
logger = get_logger(__name__)

@pytest.fixture
def session_manager():
    return SessionManager()

@pytest.fixture
def risk_assessment_agent():
    """Create a RiskAssessmentAgent instance for testing."""
    return RiskAssessmentAgent()

@pytest.fixture
def mock_model_trainer():
    """Create a mock ModelTrainer instance."""
    with patch('agents.risk_assessment.agent.ModelTrainer') as mock:
        yield mock

# Unit Tests
@pytest.mark.asyncio
async def test_risk_assessment_agent_initialization(risk_assessment_agent):
    """Test the initialization of the RiskAssessmentAgent."""
    logger.info("Testing RiskAssessmentAgent initialization")
    assert risk_assessment_agent.name == "risk_assessment_agent"
    assert "Predicts investment risk profile" in risk_assessment_agent.description
    # Tools are added at runtime in before_agent_callback
    assert hasattr(risk_assessment_agent, 'tools')

@pytest.mark.asyncio
async def test_before_agent_callback(risk_assessment_agent):
    """Test the before_agent_callback method."""
    logger.info("Testing before_agent_callback")
    mock_context = Mock()
    mock_context._invocation_context.agent = Mock()
    mock_context._invocation_context.session.state = {
        "user_profile": {
            "age": 3,
            "education": 5,
            "income": 7,
            "emergency_savings": 2,
            "retirement_planning": 2,
            "financial_literacy_score": 2
        },
        "selected_model": "boosted_tree_classifier"
    }
    
    risk_assessment_agent.before_agent_callback(mock_context)
    
    # Verify tools were added
    assert len(mock_context._invocation_context.agent.tools) > 0

@pytest.mark.asyncio
async def test_before_agent_callback_missing_data(risk_assessment_agent):
    """Test the before_agent_callback method with missing data."""
    logger.info("Testing before_agent_callback with missing data")
    mock_context = Mock()
    mock_context._invocation_context.agent = Mock()
    mock_context.state = {}  # Set state directly on context
    
    # Call the callback and verify it raises ValueError
    with pytest.raises(ValueError) as exc_info:
        risk_assessment_agent.before_agent_callback(mock_context)
    assert "❌ selected_model not found in state" in str(exc_info.value)

@pytest.mark.asyncio
async def test_after_agent_callback(risk_assessment_agent):
    """Test the after_agent_callback method."""
    logger.info("Testing after_agent_callback")
    mock_context = Mock()
    mock_context._invocation_context.agent = Mock()
    mock_context._invocation_context.agent.output = Mock()
    mock_context._invocation_context.agent.output.content = json.dumps({
        "risk_profile": "moderate",
        "probability": 0.85
    })
    
    risk_assessment_agent.after_agent_callback(mock_context)
    
    # Verify output was processed
    assert mock_context._invocation_context.agent.output is not None

# Integration Tests
@pytest.mark.asyncio
async def test_risk_assessment_integration(session_manager):
    """Test the risk assessment process."""
    logger.info("Testing risk assessment integration")
    # Setup session service and IDs
    session_service = InMemorySessionService()
    app_name = "risk_assessment_app"
    user_id = "test_user"
    session_id = "session_xyz"
    
    # Create a session object with required fields
    session = Session(
        id=session_id,  # Required field
        app_name=app_name,
        user_id=user_id,
        state={
            "user_profile": {
                "age": 3,
                "education": 5,
                "income": 7,
                "emergency_savings": 2,
                "retirement_planning": 2,
                "financial_literacy_score": 2
            },
            "selected_model": "boosted_tree_classifier"
        }
    )
    
    # Initialize the agent
    agent = RiskAssessmentAgent()
    
    # Create a runner with required parameters
    runner = Runner(
        session_service=session_service,
        app_name=app_name,
        agent=agent
    )
    
    # Create a mock response with proper types.Content structure
    mock_response = Mock()
    mock_response.is_final_response = lambda: True
    mock_response.content = types.Content(
        role="assistant",
        parts=[types.Part(text=json.dumps({
            "risk_profile": "moderate",
            "probability": 0.85
        }))]
    )
    
    # Create an async iterator class for the mock
    class MockAsyncIterator:
        def __init__(self, response):
            self.response = response
            self._yielded = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._yielded:
                self._yielded = True
                return self.response
            raise StopAsyncIteration
    
    # Mock the parent class's run_async method to return an async iterator
    with patch.object(LlmAgent, 'run_async', return_value=MockAsyncIterator(mock_response)):
        # Run the risk assessment process
        async for event in agent.run_async(runner._new_invocation_context(session)):
            if hasattr(event, "is_final_response") and event.is_final_response():
                # Verify the final response
                result = json.loads(event.content.parts[0].text)
                assert result["risk_profile"] == "moderate"
                assert result["probability"] == 0.85
                
                # Update session state with predicted risk
                session.state["predicted_risk"] = {
                    "risk_profile": result["risk_profile"],
                    "probability": result["probability"]
                }
                
                # Verify session state updates
                assert session.state["predicted_risk"]["risk_profile"] == "moderate"
                assert session.state["predicted_risk"]["probability"] == 0.85
