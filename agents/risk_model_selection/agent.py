from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from pydantic import BaseModel, Field, PrivateAttr
from typing import List, Optional
import json
import os

from agents.shared.model_utils import ModelTrainer
from utils.load_env import load_agent_env


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

    agent = callback_context._invocation_context.agent

    # Load .env for the risk_model_selection agent
    load_agent_env(__file__, "risk_model_selection")

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
        print("✅ Evaluation metrics prepared.")
        return evals_str

    agent.tools = [setup_tool, evaluate_tool]


def after_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    # agent = callback_context._invocation_context.agent
    raw_output = callback_context.state.get("selected_model")

    if not raw_output:
        print("⚠️ No selected_model found in session state.")
        return None

    print(f"📦 Raw selected_model content:\n{raw_output}")

    # Strip markdown formatting (```json ... ```)
    cleaned = raw_output.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()
    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()

    try:
        parsed = json.loads(cleaned)
        validated = ModelSelectionOutput(**parsed)

        # ✅ Log structured result
        print("✅ Structured ModelSelectionOutput:")
        print(validated.model_dump_json(indent=2))

        # ✅ Save to state (for downstream agents)
        callback_context.state["validated_result"] = validated.model_dump()

    except Exception as e:
        print(f"❌ Error parsing selected model output: {e}")

    return None  # Do not replace model output


class RiskModelSelectionAgent(LlmAgent):
    """An agent responsible for:

    1. Creating the BigQuery dataset if it doesn’t exist.
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
            output_key="selected_model",
            before_agent_callback=before_agent_callback,
            after_agent_callback=after_agent_callback,
            tools=[],  # tools added dynamically in before_agent_callback
        )

    async def run_with_runner(
        self,
        runner,
        user_id: str,
        session_id: str,
    ):
        print("🤖 Running Gemini model selection...")

        final_response = None

        user_message = types.Content(
            role="user",
            parts=[types.Part(text="Please evaluate models and select the best one.")],
        )

        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=user_message
        ):
            pass

        # Save to session state for callback access
        session = await runner.session_service.get_session(
            app_name=runner.app_name, user_id=user_id, session_id=session_id
        )

        # The callback should have parsed and saved structured result
        validated_result = session.state.get("validated_result")

        if not validated_result:
            raise RuntimeError("❌ No validated_result found in session state")

        result = ModelSelectionOutput(**validated_result)
        print(f"✅ Final response captured: {result}")

        return result.selectedModel
