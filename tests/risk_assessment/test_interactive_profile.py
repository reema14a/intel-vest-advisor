import pytest
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from agents.risk_assessment.agent import RiskAssessmentAgent


@pytest.mark.asyncio
async def test_user_profile_collection_and_prediction():
    # Setup
    session_service = InMemorySessionService()
    app_name = "risk_assessment_app"
    user_id = "interactive_test_user"
    session_id = "session_collect"

    # Only pre-set model, let agent collect user profile via tool
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state={"selected_model": "boosted_tree_classifier"},
    )

    agent = RiskAssessmentAgent()
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

    print("📋 Starting interactive user profile collection via Gradio prompts...")
    final_output = await agent.run_with_runner(runner, user_id, session_id)

    assert final_output is not None
    print("✅ Interactive Test Completed. Final Output:")
    print(final_output.parts[0].text.strip())
