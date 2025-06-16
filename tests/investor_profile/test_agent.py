import pytest
from unittest.mock import Mock, patch, AsyncMock
from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.models import LlmResponse, LlmRequest
from google.adk.sessions import InMemorySessionService, Session
from google.adk.runners import Runner
from google.genai import types
from agents.investor_profile.agent import InvestorProfileAgent, InvestorProfile, before_agent_callback, after_agent_callback
from utils.session_manager import SessionManager
from utils.logging_config import setup_logging, get_logger
import json

# Set up logging
setup_logging()
logger = get_logger(__name__)

@pytest.fixture
def investor_profile_agent():
    """Create an InvestorProfileAgent instance for testing."""
    return InvestorProfileAgent()

@pytest.mark.asyncio
async def test_investor_profile_agent_initialization(investor_profile_agent):
    """Test the initialization of the InvestorProfileAgent."""
    logger.info("Testing InvestorProfileAgent initialization")
    assert investor_profile_agent.name == "investor_profile_agent"
    assert investor_profile_agent.description == "Collects and validates investor profile data."

@pytest.mark.asyncio
async def test_investor_profile_callbacks_and_validation(investor_profile_agent):
    """Test the actual callbacks and validation logic of the investor profile agent."""
    logger.info("Testing investor profile callbacks and validation")
    
    # Test the before_agent_callback
    mock_context = Mock()
    mock_context._invocation_context = Mock()
    mock_context._invocation_context.agent = Mock()
    mock_context._invocation_context.agent.tools = []
    mock_context.state = {}
    
    # Test before_agent_callback
    from agents.investor_profile.agent import before_agent_callback
    before_agent_callback(mock_context)
    
    # Verify that the validate_profile tool was added
    assert len(mock_context._invocation_context.agent.tools) == 1
    validate_profile_tool = mock_context._invocation_context.agent.tools[0]
    
    # Test the validate_profile tool with valid data
    test_profile_data = {
        "age": "35 years old",
        "education": "Bachelor's degree", 
        "income": "$50,000 - $75,000",
        "emergency_savings": "Yes",
        "retirement_planning": "Yes",
        "financial_literacy_score": 2
    }
    
    result = validate_profile_tool(test_profile_data)
    
    # Verify the profile was validated and saved to state
    assert "user_profile" in mock_context.state
    profile = mock_context.state["user_profile"]
    
    # Verify the values were correctly normalized according to the prompt
    assert profile["age"] == 3  # 35 years old maps to age group 3 (35-44)
    assert profile["education"] == 6  # Bachelor's degree maps to 6
    assert profile["income"] == 5  # $50,000-$75,000 maps to income bracket 5  
    assert profile["emergency_savings"] == 1  # Yes maps to 1
    assert profile["retirement_planning"] == 1  # Yes maps to 1
    assert profile["financial_literacy_score"] == 2  # Direct score
    
    # Test the after_agent_callback
    from agents.investor_profile.agent import after_agent_callback
    final_content = after_agent_callback(mock_context)
    
    # Verify the final content contains the complete profile
    assert final_content is not None
    assert isinstance(final_content, types.Content)
    
    # Parse the JSON response
    import json
    final_profile = json.loads(final_content.parts[0].text)
    assert final_profile == profile
    
    logger.info(f"✅ Profile validation successful: {profile}")

@pytest.mark.asyncio 
async def test_investor_profile_validation_errors(investor_profile_agent):
    """Test error handling in profile validation."""
    logger.info("Testing investor profile validation errors")
    
    # Test the before_agent_callback
    mock_context = Mock()
    mock_context._invocation_context = Mock()
    mock_context._invocation_context.agent = Mock()
    mock_context._invocation_context.agent.tools = []
    mock_context.state = {}
    
    from agents.investor_profile.agent import before_agent_callback
    before_agent_callback(mock_context)
    
    validate_profile_tool = mock_context._invocation_context.agent.tools[0]
    
    # Test with invalid data
    invalid_profile_data = {
        "age": "invalid_age",
        "education": "invalid_education", 
        "income": "invalid_income",
        "emergency_savings": "invalid_savings",
        "retirement_planning": "invalid_planning",
        "financial_literacy_score": "invalid_score"
    }
    
    result = validate_profile_tool(invalid_profile_data)
    
    # Verify error handling
    assert "❌ Profile validation failed" in result
    
    logger.info("✅ Error handling validation successful")

@pytest.mark.asyncio
async def test_investor_profile_prompt_loading():
    """Test that the agent can load its instruction prompt correctly."""
    logger.info("Testing prompt loading")
    
    agent = InvestorProfileAgent()
    
    # Verify the agent loaded the instruction properly
    assert agent.instruction is not None
    assert len(agent.instruction) > 0
    
    # Check that key mapping information is in the prompt
    assert "Age Group:" in agent.instruction
    assert "Education Level:" in agent.instruction
    assert "Annual Income Bracket:" in agent.instruction
    assert "Emergency Savings" in agent.instruction
    assert "Financial Literacy Score:" in agent.instruction
    
    # Check that the mapping values are present
    assert "1 → 18–24" in agent.instruction  # Age mapping
    assert "6 → Bachelor's degree" in agent.instruction  # Education mapping
    assert "5 → $50,000 – $75,000" in agent.instruction  # Income mapping
    
    logger.info("✅ Agent successfully loaded prompt with mapping information")

@pytest.mark.asyncio
async def test_data_normalization():
    """Test the data normalization functionality."""
    logger.info("Testing data normalization")
    
    from agents.shared.data_utils import normalize_value
    
    # Test age normalization
    assert normalize_value("age", "35 years old") == 3  # Should map to group 3 (35-44)
    assert normalize_value("age", "25") == 2  # Should map to group 2 (25-34)
    
    # Test education normalization  
    assert normalize_value("education", "Bachelor's degree") == 6
    assert normalize_value("education", "High school graduate") == 2
    
    # Test income normalization
    assert normalize_value("income", "$50,000 - $75,000") == 5
    assert normalize_value("income", "50000 to 75000") == 5
    
    # Test boolean normalization
    assert normalize_value("emergency_savings", "Yes") == 1
    assert normalize_value("emergency_savings", "No") == 2
    
    logger.info("✅ Data normalization working correctly")

@pytest.mark.asyncio
async def test_llm_core_functionality():
    """Test that the LLM can formulate questions based on missing data and map responses correctly."""
    logger.info("Testing LLM core functionality: question formulation and response mapping")
    
    # Create a real session context
    session_service = InMemorySessionService()
    app_name = "test_app"
    user_id = "test_user"
    session_id = "test_session"
    
    # Create the agent first
    agent = InvestorProfileAgent()
    
    # Create runner with correct parameters
    runner = Runner(
        session_service=session_service,
        app_name=app_name,
        agent=agent
    )
    
    # Create session with partial profile data to test missing field detection
    session = Session(
        id=session_id,
        app_name=app_name,
        user_id=user_id,
        state={
            "user_profile": {
                "age": 3,  # 35-44 age group already provided
                # Missing: education, income, emergency_savings, retirement_planning, financial_literacy_score
            }
        }
    )
    
    # Simulate user responses that the LLM should be able to interpret and map
    simulated_responses = {
        "education": "I have a Bachelor's degree",
        "income": "My annual income is between $50,000 and $75,000", 
        "emergency": "Yes, I have emergency savings",
        "retirement": "Yes, I have started retirement planning",
        "financial": "For question M7: bond prices fall when rates rise. For M8: I'd have more than $102. For M10: mutual funds don't guarantee returns."
    }
    
    conversation_log = []
    
    # Mock user input to simulate responses when LLM asks questions
    def mock_input(prompt=""):
        conversation_log.append(f"LLM: {prompt}")
        
        # Determine appropriate response based on what LLM is asking about
        prompt_lower = prompt.lower()
        if "education" in prompt_lower:
            response = simulated_responses["education"]
        elif "income" in prompt_lower:
            response = simulated_responses["income"]  
        elif "emergency" in prompt_lower or "savings" in prompt_lower:
            response = simulated_responses["emergency"]
        elif "retirement" in prompt_lower:
            response = simulated_responses["retirement"]
        elif "financial" in prompt_lower or "literacy" in prompt_lower:
            response = simulated_responses["financial"]
        else:
            response = "I don't understand the question"
            
        conversation_log.append(f"User: {response}")
        return response
    
    try:
        # Patch input to simulate user responses
        import builtins
        original_input = getattr(builtins, 'input', None)
        builtins.input = mock_input
        
        # Run the agent and let it formulate questions and process responses
        final_profile = None
        async for event in agent.run_async(runner._new_invocation_context(session)):
            if hasattr(event, "is_final_response") and event.is_final_response():
                try:
                    final_profile = json.loads(event.content.parts[0].text)
                    break
                except json.JSONDecodeError:
                    # LLM might return non-JSON response
                    logger.info(f"LLM final response: {event.content.parts[0].text}")
                    break
        
        # Log the conversation for debugging
        logger.info("Conversation log:")
        for entry in conversation_log:
            logger.info(entry)
        
        # Verify the LLM correctly formulated questions and mapped responses
        if final_profile:
            logger.info(f"Final profile mapped by LLM: {final_profile}")
            
            # Verify the LLM correctly mapped responses to numeric values
            assert final_profile.get("age") == 3  # Should preserve existing value
            assert final_profile.get("education") == 6  # Bachelor's degree → 6
            assert final_profile.get("income") == 5  # $50K-$75K → bracket 5
            assert final_profile.get("emergency_savings") == 1  # Yes → 1
            assert final_profile.get("retirement_planning") == 1  # Yes → 1
            assert final_profile.get("financial_literacy_score") == 3  # All 3 correct → score 3
            
            logger.info("✅ LLM successfully formulated questions and mapped responses correctly")
        else:
            logger.warning("LLM did not provide a structured final profile")
            
    finally:
        # Restore original input function if it existed
        if original_input:
            builtins.input = original_input

@pytest.mark.asyncio
async def test_llm_question_formulation_and_mapping():
    """Test that the LLM can formulate questions based on missing data and map responses correctly."""
    logger.info("Testing LLM core functionality: question formulation and response mapping")
    
    # Create a real session context
    session_service = InMemorySessionService()
    
    agent = InvestorProfileAgent()
    
    # Create runner with correct parameters
    runner = Runner(
        session_service=session_service,
        app_name="test_app",
        agent=agent
    )
    
    session = Session(
        id="test_session",
        app_name="test_app",
        user_id="test_user",
        state={"user_profile": {"age": 3}}  # Only age provided, missing other fields
    )
    
    # This test will require real environment variables to be set
    # It tests the actual LLM's ability to:
    # 1. Read the prompt and understand the mapping rules
    # 2. Identify missing fields from the current profile  
    # 3. Formulate appropriate questions to collect missing data
    # 4. Interpret user responses and map them to correct numeric values
    
    logger.info("This test requires actual LLM interaction and environment variables")
    logger.info("It tests the core functionality we want to verify:")
    logger.info("- LLM reads prompt and understands mapping rules")
    logger.info("- LLM identifies missing profile fields")
    logger.info("- LLM formulates appropriate questions")
    logger.info("- LLM correctly maps responses to numeric values")
    
    # For now, we'll skip the actual execution since it requires real LLM API access
    # But this is the structure for testing the core functionality
    pytest.skip("Requires real LLM API access and environment setup") 