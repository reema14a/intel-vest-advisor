import logging
from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from pydantic import BaseModel, Field, PrivateAttr
from typing import List, Optional, AsyncGenerator
import json
import os

from agents.shared.model_utils import ModelTrainer
from utils.load_env import load_agent_env
from utils.monitoring import monitoring

# Configure logging
logger = logging.getLogger(__name__)


class ModelMetric(BaseModel):
    model_name: str = Field(description="Name of the evaluated model")
    recall: float = Field(description="Recall score for the model")
    f1_score: float = Field(description="F1 score for the model")
    precision: float = Field(description="Precision score for the model")
    accuracy: float = Field(description="Accuracy score for the model")


class ModelSelectionOutput(BaseModel):
    selectedModel: str = Field(description="Name of the selected model")
    reason: str = Field(description="Reason for selecting the model")
    metrics: List[ModelMetric] = Field(
        description="List of evaluation metrics for each model"
    )


def before_agent_callback(callback_context: CallbackContext):
    """Dynamically load environment and inject tools"""
    logger.info("Starting before_agent_callback for risk model selection")

    agent = callback_context._invocation_context.agent

    # Load .env for the risk_model_selection agent
    load_agent_env(__file__, "risk_model_selection")
    logger.debug("Environment variables loaded for risk model selection agent")

    # Validate and extract
    project_id = os.getenv("GCP_PROJECT_ID")
    dataset_id = os.getenv("BQ_DATASET")
    table_name = os.getenv("BQ_TABLE")
    data_path = os.getenv("DATA_PATH")
    sql_base = os.getenv("SQL_BASE")

    assert all(
        [project_id, dataset_id, table_name, data_path, sql_base]
    ), "❌ Missing one or more required env variables"

    # Create trainer functions and inject
    def setup_tool():
        """Set up BigQuery dataset and train models."""
        trainer = ModelTrainer(project_id, dataset_id, sql_base, table_name, data_path)
        trainer.train_all_models()
        return "Setup and model training completed."

    def evaluate_tool():
        """Evaluate trained models and return metrics."""
        trainer = ModelTrainer(project_id, dataset_id, sql_base, table_name, data_path)
        evaluations = trainer.evaluate_all_models()

        if not evaluations:
            return "⚠️ No trained models found. You must call `setup_tool` first."

        evals_str = "\n".join(
            f"{row['model_name']}: Recall={row['recall']:.3f}, F1={row['f1_score']:.3f}, "
            f"Precision={row['precision']:.3f}, Accuracy={row['accuracy']:.3f}"
            for row in evaluations
        )
        logger.debug("✅ Evaluation metrics prepared.")
        return evals_str

    agent.tools = [setup_tool, evaluate_tool]


def after_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    # Get the selected model output from session state, not local callback context state
    session_state = callback_context._invocation_context.session.state
    
    # Try multiple sources for the raw output
    raw_output = (
        session_state.get("model_selection_raw_output") or
        session_state.get("risk_model_selection_agent_output") or
        ""
    )

    # If we still don't have raw output, try to get it from the agent's last response
    if not raw_output:
        # Check if there's a recent response we can extract JSON from
        logger.warning("⚠️ No raw output found in session state, checking recent responses...")
        
        # The LLM response should be in the session state somewhere
        for key, value in session_state.items():
            if isinstance(value, str) and "selectedModel" in value and "```json" in value:
                raw_output = value
                logger.debug(f"📦 Found JSON in session state key: {key}")
                break

    if not raw_output:
        logger.warning("⚠️ No JSON output found anywhere in session state.")
        logger.debug(f"Session state keys: {list(session_state.keys())}")
        return None

    logger.debug(f"📦 Raw selected_model content:\n{raw_output}")

    # Extract JSON from the response text
    json_start = raw_output.find("```json")
    json_end = raw_output.find("```", json_start + 7)
    
    if json_start != -1 and json_end != -1:
        # Extract the JSON part
        json_text = raw_output[json_start + 7:json_end].strip()
    else:
        # Try to find JSON without markdown formatting
        json_start = raw_output.find("{")
        json_end = raw_output.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_text = raw_output[json_start:json_end]
        else:
            logger.error("❌ Could not find JSON in the output")
            return None

    try:
        parsed = json.loads(json_text)
        validated = ModelSelectionOutput(**parsed)

        # ✅ Log structured result
        logger.debug("✅ Structured ModelSelectionOutput:")
        logger.debug(validated.model_dump_json(indent=2))

        # ✅ Save to session state (for downstream agents)
        # Store both the full result and just the selected model name for easy access
        session_state["validated_result"] = validated.model_dump()
        session_state["selected_model"] = validated.selectedModel
        
        logger.info(f"✅ Successfully saved selected_model: {validated.selectedModel}")

    except Exception as e:
        logger.error(f"❌ Error parsing selected model output: {e}")
        logger.error(f"JSON text that failed to parse: {json_text}")

    return None  # Do not replace model output


class RiskModelSelectionAgent(LlmAgent):
    """An agent responsible for:

    1. Creating the BigQuery dataset if it doesn't exist.
    2. Uploading cleaned training data (from Parquet).
    3. Evaluating multiple model types (`LOGISTIC_REG`, `BOOSTED_TREE_CLASSIFIER`, `DNN_CLASSIFIER`, etc.).
    4. Selecting the best model based on evaluation metrics.
    5. Training and storing the final model for use by other agents."""

    # Declare private attribute
    _validated_result: ModelSelectionOutput | None = PrivateAttr(default=None)

    def _get_instruction(self):
        instruction = """
        You are a machine learning expert. You are evaluating models trained to classify investor risk profiles (Conservative, Moderate, Aggressive).
        Decide whether you need to call `setup_tool` to retrain models (if you suspect drift or major data change)
        or if you can use `evaluate_tool` to work with the latest stored metrics.
        You must call `evaluate_tool` to retrieve model evaluation metrics.
        If it returns **No trained models found. You must call `setup_tool` first.**, 
        immediately call `setup_tool` to train the models, and then call `evaluate_tool` again.
        Focus on selecting the model with the best balance of Recall and F1 score.
        
        Important: 
        Do not proceed to return your final JSON unless you have a valid list of model metrics.
        Return your final answer strictly in this JSON format:
        {
          "selectedModel": "<model_name>",
          "reason": "<why you selected it, including drift or not>",
          "metrics": [
            {
              "model_name": "<model>",
              "recall": <value>,
              "f1_score": <value>,
              "precision": <value>,
              "accuracy": <value>
            },
            ...
          ]
        }
        """
        return instruction

    def __init__(self):
        # === Define local-wrapped tool functions ===
        super().__init__(
            model=os.getenv("MODEL_NAME", "gemini-2.0-flash"),
            name="risk_model_selection_agent",
            description="Selects the best BigQuery ML model based on evaluation metrics.",
            instruction=self._get_instruction(),
            output_key="model_selection_raw_output",  # Use different key to avoid conflicts
            before_agent_callback=before_agent_callback,
            after_agent_callback=after_agent_callback,
            tools=[],  # tools added dynamically in before_agent_callback
        )

    async def run_async(self, ctx: CallbackContext) -> AsyncGenerator[Event, None]:
        """Run the model selection workflow asynchronously."""
        # Just run the parent's run_async, don't inject a synthetic message
        async for event in super().run_async(ctx):
            if hasattr(event, "is_final_response") and event.is_final_response():
                print(f"✅ Model Selection Complete:\n{event.content.parts[0].text.strip()}")
            yield event
            
        # After model selection completes, create a final message event
        selected_model = ctx.session.state.get("selected_model")
        if not selected_model:
            yield Event(
                author="risk_model_selection_agent",
                content=types.Content(
                    role="assistant",
                    parts=[types.Part(text="Error: No model was selected")]
                )
            )
            return
            
        # Create final message with selected model
        yield Event(
            author="risk_model_selection_agent",
            content=types.Content(
                role="assistant",
                parts=[types.Part(text=f"Selected model: {selected_model}")]
            )
        )
