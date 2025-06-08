import pytest
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from agents.risk_assessment.agent import RiskAssessmentAgent


@pytest.mark.asyncio
async def test_risk_assessment_integration():
    session_service = InMemorySessionService()
    app_name = "risk_assessment_app"
    user_id = "test_user"
    session_id = "session_xyz"

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state={
            "selected_model": "boosted_tree_classifier",
            "user_profile": {
                "age": 3,
                "education": 4,
                "income": 5,
                "emergency_savings": 2,
                "retirement_planning": 2,
                "financial_literacy_score": 2,
            },
        },
    )

    agent = RiskAssessmentAgent()
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)
    final_output = await agent.run_with_runner(runner, user_id, session_id)

    assert final_output is not None
    print("✅ Test Completed. Final Output:")
    print(final_output.parts[0].text.strip())
