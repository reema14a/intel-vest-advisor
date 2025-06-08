import pytest

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

from agents.risk_model_selection.agent import (
    RiskModelSelectionAgent,
    ModelSelectionOutput,
)


@pytest.mark.asyncio
async def test_risk_model_selection_integration():
    # Setup session service and IDs
    session_service = InMemorySessionService()
    app_name = "risk_model_app"
    user_id = "test_user"
    session_id = "session_xyz"
    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )

    # Initialize the agent
    agent = RiskModelSelectionAgent()

    # Create runner
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

    # Run the agent with session details
    result: ModelSelectionOutput = await agent.run_with_runner(
        runner=runner,
        user_id=user_id,
        session_id=session_id,
    )

    # Validate selected model against allowed list
    allowed_models = ["LOGISTIC_REG", "BOOSTED_TREE_CLASSIFIER", "DNN_CLASSIFIER"]
    assert result.upper() in allowed_models, f"Unexpected selected model: {result}"
