import threading
from google.adk.agents import LlmAgent
from google.adk.models import LlmResponse, LlmRequest
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from pydantic import BaseModel, Field, PrivateAttr
from typing import Optional, Literal
import os
import json
import gradio as gr
import time

from agents.shared.data_utils import normalize_value
from agents.shared.model_utils import ModelTrainer
from utils.load_env import load_agent_env

PROMPT_FILE = "agents/prompts/user_profile_prompt_with_questions.txt"


class RiskAssessmentInput(BaseModel):
    age: Literal[1, 2, 3, 4, 5, 6] = Field(
        ..., description="Age group: 1=18–24, ..., 6=65+"
    )
    education: Literal[1, 2, 3, 4, 5, 6] = Field(
        ..., description="Education level: 1=No schooling, ..., 6=Postgrad"
    )
    income: int = Field(
        ..., description="Income group on a scale of 1 (lowest) to 10 (highest)"
    )
    emergency_savings: Literal[1, 2, 3] = Field(
        ..., description="Emergency savings level: 1=none, 2=some, 3=adequate"
    )
    retirement_planning: Literal[1, 2, 3] = Field(
        ..., description="Retirement planning: 1=none, 2=some, 3=active"
    )
    financial_literacy_score: int = Field(
        ..., description="Score from 0 to 3 based on financial knowledge"
    )


class RiskAssessmentOutput(BaseModel):
    risk_profile: str = Field(..., description="Predicted risk profile")
    probability: float = Field(..., description="Model confidence for this profile")


def before_agent_callback(callback_context: CallbackContext):
    """Inject predict_tool into the agent using selected model."""
    state = callback_context.state

    # Load .env for the risk_assessment agent
    load_agent_env(__file__, "risk_assessment")

    print("Before Agent callback started")

    project_id = os.getenv("GCP_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET")
    sql_base = os.getenv("SQL_BASE")

    selected_model = state.get("selected_model")

    if not selected_model:
        raise ValueError("❌ selected_model not found in session state")

    trainer = ModelTrainer(
        project_id=project_id, dataset_id=dataset_id, sql_base=sql_base
    )

    def predict_tool(
        age: int,
        education: int,
        income: int,
        emergency_savings: int,
        retirement_planning: int,
        financial_literacy_score: int,
    ):
        try:
            # Normalize
            inputs = {
                "age": normalize_value("age", age),
                "education": normalize_value("education", education),
                "income": normalize_value("income", income),
                "emergency_savings": normalize_value(
                    "emergency_savings", emergency_savings
                ),
                "retirement_planning": normalize_value(
                    "retirement_planning", retirement_planning
                ),
                "financial_literacy_score": financial_literacy_score,
            }
            validated = RiskAssessmentInput(**inputs)
            prediction = trainer.predict_risk_profile(
                model_name=selected_model,
                **validated.model_dump(),
            )
            return json.dumps(prediction, indent=2)
        except Exception as e:
            return f"❌ Error during prediction: {str(e)}"

    def ask_user_input(prompt_text: str) -> str:
        result = {}

        def submit_fn(input_value):
            result["value"] = input_value
            return f"Thanks! You entered: {input_value}"

        def launch_ui():
            with gr.Blocks() as demo:
                gr.Markdown(f"### {prompt_text}")
                user_input = gr.Textbox()
                submit_btn = gr.Button("Submit")
                output = gr.Textbox()

                submit_btn.click(fn=submit_fn, inputs=user_input, outputs=output)

            demo.launch(
                inbrowser=True,
                prevent_thread_lock=True,
                share=False,
                enable_queue=False,
            )  # 👈 important

        # Launch Gradio UI in a separate thread
        ui_thread = threading.Thread(target=launch_ui)
        ui_thread.start()

        # Wait for user input to be stored
        while "value" not in result:
            time.sleep(0.1)

        return result["value"]

    callback_context._invocation_context.agent.tools = [predict_tool, ask_user_input]


def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest):
    """Check which fields are missing and ask user for them."""
    print("Before Model callback started")

    state = callback_context.state
    collected = state.get("user_profile", {})

    required_fields = RiskAssessmentInput.model_fields.keys()
    for field in required_fields:
        if field not in collected:
            callback_context.prompt = f"Please provide your {field.replace('_', ' ')}:"
            return  # triggers multi-turn


def after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse):
    """Normalize field values."""

    print("After Agent callback started")

    state = callback_context.state
    if "user_profile" not in state:
        return

    profile = state["user_profile"]
    for field in [
        "age",
        "education",
        "income",
        "emergency_savings",
        "retirement_planning",
    ]:
        profile[field] = normalize_value(field, profile.get(field, 99))


def after_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    print("After Agent callback started")

    state = callback_context.state.to_dict()
    user_data = state.get("user_profile")
    selected_model = state.get("selected_model")

    if not user_data or not selected_model:
        return types.Content(parts=[types.Part(text="Missing input or model.")])

    try:
        input_obj = RiskAssessmentInput(**user_data)
        trainer = ModelTrainer(
            project_id=os.getenv("GCP_PROJECT_ID"),
            dataset_id=os.getenv("BQ_DATASET"),
            sql_base=os.getenv("SQL_BASE"),
        )
        result = trainer.predict_risk_profile(
            f"{selected_model}", **input_obj.model_dump()
        )

        if not result:
            return types.Content(parts=[types.Part(text="No prediction available.")])

        output = RiskAssessmentOutput(**result)
        callback_context.state["predicted_risk"] = output.model_dump()
        return types.Content(
            parts=[types.Part(text=json.dumps(output.model_dump(), indent=2))]
        )

    except Exception as e:
        return types.Content(parts=[types.Part(text=f"Prediction failed: {e}")])


def load_instruction():
    with open(PROMPT_FILE) as f:
        return f.read()


class RiskAssessmentAgent(LlmAgent):
    _user_input: Optional[RiskAssessmentInput] = PrivateAttr(default=None)

    def __init__(self):
        super().__init__(
            name="risk_assessment_agent",
            description="Collects user profile and predicts investment risk profile using trained model.",
            model=os.getenv("MODEL_NAME", "gemini-2.0-flash"),
            instruction=load_instruction(),
            before_agent_callback=before_agent_callback,
            before_model_callback=before_model_callback,
            after_model_callback=after_model_callback,
            after_agent_callback=after_agent_callback,
            tools=[],
            output_key="predicted_risk",  # predict_tool added at runtime
        )

    async def run_with_runner(self, runner, user_id: str, session_id: str):
        message = types.Content(
            role="user",
            parts=[types.Part(text="Let's start collecting user profile.")],
        )

        final_output = None
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=message
        ):
            if hasattr(event, "is_final_response") and event.is_final_response():
                final_output = event.content
                print(f"✅ Final Risk Profile:\n{final_output.parts[0].text.strip()}")
        return final_output
