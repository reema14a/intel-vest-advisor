import pytest
import asyncio
from unittest.mock import Mock, patch
from google.adk.sessions import InMemorySessionService, Session
from google.adk.runners import Runner
from agents.orchestration.agent import OrchestrationAgent, ConversationState
from agents.risk_assessment.agent import RiskAssessmentAgent
from agents.risk_model_selection.agent import RiskModelSelectionAgent
from agents.investor_profile.agent import InvestorProfileAgent, InvestorProfile
from google.genai import types
import json
from utils.session_manager import SessionManager

@pytest.fixture
def mock_runner():
    runner = Mock()
    runner.state = {}
    return runner

@pytest.fixture
def session_manager():
    return SessionManager()

@pytest.fixture
def orchestration_agent(session_manager):
    return OrchestrationAgent(session_manager)

@pytest.fixture
def risk_assessment_agent():
    return RiskAssessmentAgent()

@pytest.fixture
def risk_model_selection_agent():
    return RiskModelSelectionAgent()

@pytest.fixture
def investor_profile_agent():
    return InvestorProfileAgent()

@pytest.mark.asyncio
async def test_orchestration_agent_initialization(orchestration_agent):
    assert orchestration_agent.name == "orchestration_agent"
    assert orchestration_agent.description == "Orchestrates the investment advisory process through a sequential workflow"
    # Verify it's a Custom Agent with correct sub-agents
    assert hasattr(orchestration_agent, 'sub_agents')
    assert len(orchestration_agent.sub_agents) == 3
    
    # Check that the sub-agents are of the correct types
    investor_profile_agent = orchestration_agent.sub_agents[0]
    risk_assessment_agent = orchestration_agent.sub_agents[1]
    risk_model_selection_agent = orchestration_agent.sub_agents[2]
    
    assert isinstance(investor_profile_agent, InvestorProfileAgent)
    assert isinstance(risk_assessment_agent, RiskAssessmentAgent)
    assert isinstance(risk_model_selection_agent, RiskModelSelectionAgent)
    
    # Verify agent names
    assert investor_profile_agent.name == "investor_profile_agent"
    assert risk_assessment_agent.name == "risk_assessment_agent"
    assert risk_model_selection_agent.name == "risk_model_selection_agent"

@pytest.mark.asyncio
async def test_conversation_state_initialization(orchestration_agent, mock_runner):
    # Test initial state
    state = ConversationState()
    assert state.current_agent == "orchestration"
    assert state.collected_data == {}
    assert state.next_steps == []
    assert state.completed_steps == []

@pytest.mark.asyncio
async def test_error_handling(orchestration_agent, mock_runner):
    # Mock error in _run_async_impl
    async def mock_error_run_async_impl(ctx):
        raise Exception("Test error")
    
    with patch.object(orchestration_agent, '_run_async_impl', mock_error_run_async_impl):
        # Start conversation
        response = await orchestration_agent.run_with_runner(
            mock_runner,
            "test_user",
            "test_session"
        )
        
        # Verify error handling
        assert response is not None
        assert "error" in response.parts[0].text.lower()

@pytest.mark.asyncio
async def test_final_recommendations(orchestration_agent):
    # Test the recommendation generation directly
    risk_profile = {
        "risk_profile": "moderate",
        "probability": 0.85
    }
    selected_model = {
        "selectedModel": "balanced_portfolio",
        "reason": "Best fit for moderate risk tolerance"
    }
    
    # Generate recommendations
    recommendations = orchestration_agent._generate_final_recommendations(risk_profile, selected_model)
    
    # Verify recommendations content
    assert "moderate" in recommendations.lower()
    assert "balanced_portfolio" in recommendations.lower()
    assert "85.0%" in recommendations  # Confidence percentage
    assert "Investment Strategy Recommendations" in recommendations
    assert "Important Disclaimer" in recommendations

def test_ui_initialization():
    """Test UI initialization - simplified to avoid import conflicts"""
    # Since we fixed the import issue, let's test that the import works
    try:
        from ui.client import InvestmentAdvisorUI
        # Just test that we can import the class
        assert InvestmentAdvisorUI is not None
    except ImportError as e:
        pytest.fail(f"Failed to import InvestmentAdvisorUI: {e}")

@pytest.mark.asyncio
async def test_ui_compatibility(session_manager):
    """Test that UI components work with the orchestration agent"""
    # Test that the orchestration agent is compatible with UI expectations
    from ui.investment_advisor_ui import InvestmentAdvisorUI
    
    # Create UI instance
    ui = InvestmentAdvisorUI(session_manager)
    
    # Verify the agent is the correct type
    assert isinstance(ui.orchestrator, OrchestrationAgent)
    assert hasattr(ui.orchestrator, 'run_with_runner')  # Required for UI compatibility 

@pytest.mark.asyncio
async def test_orchestration_workflow(orchestration_agent, mock_runner, session_manager):
    # Mock the sub-agents' run_async methods
    async def mock_investor_profile_run(ctx):
        profile_data = {
            "age": 30,
            "education": 5,
            "income": 7,
            "emergency_savings": 2,
            "retirement_planning": 2,
            "financial_literacy_score": 2
        }
        ctx.state["user_profile"] = profile_data
        yield Mock(
            is_final_response=lambda: True,
            content=types.Content(parts=[types.Part(text=json.dumps(profile_data))])
        )

    async def mock_risk_assessment_run(ctx):
        risk_profile = {
            "risk_profile": "moderate",
            "probability": 0.85
        }
        ctx.state["predicted_risk"] = risk_profile
        yield Mock(
            is_final_response=lambda: True,
            content=types.Content(parts=[types.Part(text=json.dumps(risk_profile))])
        )

    async def mock_model_selection_run(ctx):
        selected_model = {
            "selectedModel": "balanced_portfolio",
            "reason": "Best fit for moderate risk tolerance"
        }
        ctx.state["validated_result"] = selected_model
        yield Mock(
            is_final_response=lambda: True,
            content=types.Content(parts=[types.Part(text=json.dumps(selected_model))])
        )

    # Create a session object with required fields
    session = Session(
        id="test_session",  # Required field
        app_name="test_app",
        user_id="test_user",
        state={}
    )

    # Patch the sub-agents' run_async methods
    with patch.object(orchestration_agent.sub_agents[0], 'run_async', mock_investor_profile_run), \
         patch.object(orchestration_agent.sub_agents[1], 'run_async', mock_risk_assessment_run), \
         patch.object(orchestration_agent.sub_agents[2], 'run_async', mock_model_selection_run):
        
        # Run the orchestration workflow
        async for event in orchestration_agent.run_async(mock_runner._new_invocation_context(session)):
            if hasattr(event, "is_final_response") and event.is_final_response():
                # Verify the final response contains recommendations
                assert "balanced_portfolio" in event.content.parts[0].text
                assert "moderate risk tolerance" in event.content.parts[0].text
                
                # Verify session state updates
                assert session.state["user_profile"] is not None
                assert session.state["predicted_risk"] is not None
                assert session.state["validated_result"] is not None

@pytest.mark.asyncio
async def test_orchestration_workflow_failure(orchestration_agent, mock_runner):
    # Mock a failure in the investor profile collection
    async def mock_failed_profile_run(ctx):
        raise Exception("Failed to collect profile")
    
    # Create a session object with required fields
    session = Session(
        id="test_session",  # Required field
        app_name="test_app",
        user_id="test_user",
        state={}
    )
    
    # Patch the first sub-agent to fail
    with patch.object(orchestration_agent.sub_agents[0], 'run_async', mock_failed_profile_run):
        # Run the orchestration workflow
        async for event in orchestration_agent.run_async(mock_runner._new_invocation_context(session)):
            if hasattr(event, "is_final_response") and event.is_final_response():
                # Verify error handling
                assert "Failed to collect profile" in event.content.parts[0].text
                assert session.state.get("error") is not None

@pytest.mark.asyncio
async def test_conversation_state_management(orchestration_agent, mock_runner):
    # Test initial state
    state = ConversationState()
    assert state.current_agent == "orchestration"
    assert state.collected_data == {}
    assert state.next_steps == []
    assert state.completed_steps == []
    
    # Test state updates
    state.current_agent = "investor_profile"
    state.collected_data = {"user_profile": {"age": 3}}
    state.next_steps = ["risk_assessment"]
    state.completed_steps = ["investor_profile"]
    
    assert state.current_agent == "investor_profile"
    assert state.collected_data["user_profile"]["age"] == 3
    assert "risk_assessment" in state.next_steps
    assert "investor_profile" in state.completed_steps 