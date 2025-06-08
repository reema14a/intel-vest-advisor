import asyncio
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from agents.risk_assessment.agent import RiskAssessmentAgent


async def main():
    app_name = "risk_assessment_app"
    user_id = "test_user"
    session_id = "session_gradio"

    # Only selected_model is set; user_profile will be collected via agent interaction
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state={"selected_model": "boosted_tree_classifier"},
    )

    agent = RiskAssessmentAgent()
    runner = Runner(agent=agent, app_name=app_name, session_service=session_service)

    final_output = await agent.run_with_runner(runner, user_id, session_id)

    if final_output:
        print("✅ Final Output:")
        print(final_output.parts[0].text.strip())
    else:
        print("❌ No final output returned.")


if __name__ == "__main__":
    asyncio.run(main())
