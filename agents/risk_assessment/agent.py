import logging
from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.models import LlmResponse, LlmRequest
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from pydantic import BaseModel, Field, PrivateAttr
from typing import Optional, Literal, AsyncGenerator
import os
import json

from agents.shared.model_utils import ModelTrainer
from utils.load_env import load_agent_env
from utils.monitoring import monitoring

# Configure logging
logger = logging.getLogger(__name__)

class RiskAssessmentOutput(BaseModel):
    risk_profile: str = Field(..., description="Predicted risk profile")
    probability: float = Field(..., description="Model confidence for this profile")

def before_agent_callback(callback_context: CallbackContext):
    """Inject predict_tool into the agent using selected model."""
    logger.info("Starting before_agent_callback for risk assessment")
    
    # Get data from session state instead of local callback context state
    session_state = callback_context._invocation_context.session.state

    # Load .env for the risk_assessment agent
    load_agent_env(__file__, "risk_assessment")
    logger.debug("Environment variables loaded for risk assessment agent")

    project_id = os.getenv("GCP_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET")
    sql_base = os.getenv("SQL_BASE")

    selected_model = session_state.get("selected_model")
    user_profile = session_state.get("user_profile")

    if not selected_model:
        raise ValueError("❌ selected_model not found in session state")
    if not user_profile:
        raise ValueError("❌ user_profile not found in session state")

    trainer = ModelTrainer(
        project_id=project_id, dataset_id=dataset_id, sql_base=sql_base
    )

    def predict_tool():
        try:
            prediction = trainer.predict_risk_profile(
                model_name=selected_model,
                **user_profile
            )
            # Save prediction to both states
            callback_context.state["predicted_risk"] = prediction
            callback_context._invocation_context.session.state["predicted_risk"] = prediction
            return json.dumps(prediction, indent=2)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return f"❌ Error during prediction: {str(e)}"

    callback_context._invocation_context.agent.tools = [predict_tool]

def after_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    """Process final results and save to both states."""
    logger.info("Starting after agent callback")

    state = callback_context.state
    prediction = state.get("predicted_risk")

    if not prediction:
        logger.error("No prediction available")
        return types.Content(parts=[types.Part(text="No prediction available.")])

    try:
        # Ensure prediction is a dictionary
        if isinstance(prediction, str):
            try:
                prediction = json.loads(prediction)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse prediction string: {prediction}")
                return types.Content(parts=[types.Part(text="Error: Invalid prediction format.")])

        # Create output object
        output = RiskAssessmentOutput(**prediction)
        
        # Save to both states
        output_dict = output.model_dump()
        callback_context.state["predicted_risk"] = output_dict
        callback_context._invocation_context.session.state["predicted_risk"] = output_dict
        
        return types.Content(
            parts=[types.Part(text=json.dumps(output_dict, indent=2))]
        )

    except Exception as e:
        logger.error(f"Prediction processing failed: {str(e)}")
        return types.Content(parts=[types.Part(text=f"Prediction processing failed: {e}")])

class RiskAssessmentAgent(LlmAgent):
    def __init__(self):
        super().__init__(
            name="risk_assessment_agent",
            description="Predicts investment risk profile using trained model.",
            model=os.getenv("MODEL_NAME", "gemini-2.0-flash"),
            instruction="Predict the risk profile based on the provided investor profile data.",
            before_agent_callback=before_agent_callback,
            after_agent_callback=after_agent_callback,
            tools=[], # predict_tool added at runtime
            output_key="predicted_risk",
        )

    async def run_async(self, ctx: CallbackContext) -> AsyncGenerator[Event, None]:
        """Run the risk assessment workflow asynchronously."""
        message = types.Content(
            role="user",
            parts=[types.Part(text="Let's assess your risk profile.")],
        )

        async for event in super().run_async(ctx):
            if hasattr(event, "is_final_response") and event.is_final_response():
                logger.info(f"✅ Final Risk Profile:\n{event.content.parts[0].text.strip()}")
            yield event
