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

from agents.shared.data_utils import normalize_value
from utils.load_env import load_agent_env
from utils.monitoring import monitoring

# Configure logging
logger = logging.getLogger(__name__)

PROMPT_FILE = "agents/prompts/user_profile_prompt_with_questions.txt"

# Module-level state to persist data across tool calls within the same agent run
_current_profile = {}

# def get_current_profile(callback_context: CallbackContext):
#     """Get the current profile from session state."""
#     return callback_context._invocation_context.session.state.get("user_profile_partial", {})

# def save_profile(callback_context: CallbackContext, profile_data):
#     """Save profile to session state."""
#     callback_context._invocation_context.session.state["user_profile_partial"] = profile_data
#     # If complete, also save to main profile key
#     missing = [f for f in InvestorProfile.model_fields.keys() if f not in profile_data]
#     if not missing:
#         callback_context._invocation_context.session.state["user_profile"] = profile_data

class InvestorProfile(BaseModel):
    age: Literal[1, 2, 3, 4, 5, 6] = Field(
        ..., description="Age group: 1=18–24, ..., 6=65+"
    )
    education: Literal[1, 2, 3, 4, 5, 6, 7, -1] = Field(
        ..., description="Education level: 1=No schooling, ..., 7=Postgrad, -1=Unknown/Prefer not to say"
    )
    income: int = Field(
        ..., description="Income group on a scale of 1 (lowest) to 10 (highest)"
    )
    emergency_savings: Literal[1, 2] = Field(
        ..., description="Emergency savings: 1=Yes, 2=No"
    )
    retirement_planning: Literal[1, 2, 3] = Field(
        ..., description="Retirement planning: 1=Yes, 2=No, 3=Unknown"
    )
    financial_literacy_score: int = Field(
        ..., description="Score from 0 to 3 based on financial knowledge"
    )

def before_agent_callback(callback_context: CallbackContext):
    """Set up tools for profile collection and validation."""
    logger.info("Starting before_agent_callback for investor profile collection")

    def update_profile(profile_data: dict) -> str:
        """Update profile data with one or more fields."""
        try:
            logger.info(f"update_profile called with: {profile_data}")
            
            # According to ADK docs, Custom Agents manage state in session.state
            # Sub-agents should write to session state directly for persistence
            existing_profile = callback_context._invocation_context.session.state.get("user_profile_partial", {}).copy()
            logger.info(f"Existing partial profile from session state: {existing_profile}")
            
            # Update with new data
            updated_fields = []
            for key, value in profile_data.items():
                if value is not None:
                    # Handle all fields consistently
                    if key in ["age", "education", "income", "emergency_savings", "retirement_planning"]:
                        normalized_value = normalize_value(key, value)
                        existing_profile[key] = normalized_value
                        updated_fields.append(f"{key}: {value} -> {normalized_value}")
                        logger.info(f"Stored {key}: {value} -> {normalized_value}")
                    elif key == "financial_literacy_score":
                        # Handle financial_literacy_score which might come as [score] or score
                        if isinstance(value, list) and len(value) > 0:
                            score = value[0]  # Extract from list
                        else:
                            score = value  # Use directly if not a list
                        existing_profile[key] = int(score)
                        updated_fields.append(f"{key}: {value} -> {score}")
                        logger.info(f"Stored {key}: {value} -> {score}")
                    else:
                        # For any other unexpected fields
                        existing_profile[key] = value
                        updated_fields.append(f"{key}: {value}")
                        logger.info(f"Stored {key}: {value}")
            
            # Save directly to session state (persists across agent calls)
            callback_context._invocation_context.session.state["user_profile_partial"] = existing_profile
            logger.info(f"Saved partial profile to session state: {existing_profile}")
            
            # Also save to database for reliable persistence
            try:
                from database.session_store import DatabaseSessionStore
                db_store = DatabaseSessionStore()
                session_id = getattr(callback_context._invocation_context.session, 'id', 'unknown_session')
                db_store.create_or_update_user_profile(session_id, existing_profile)
                logger.info(f"✅ Saved partial profile to database: {existing_profile}")
            except Exception as e:
                logger.error(f"❌ Failed to save profile to database: {str(e)}")
            
            # If complete, also save as final user_profile
            missing_check = [f for f in InvestorProfile.model_fields.keys() if f not in existing_profile]
            if not missing_check:
                callback_context._invocation_context.session.state["user_profile"] = existing_profile
                logger.info("✅ Complete profile saved to user_profile in session state")
            
            # Check completion
            missing = [f for f in InvestorProfile.model_fields.keys() if f not in existing_profile]
            
            if missing:
                response = f"✅ Got it! Updated: {', '.join(updated_fields)}. Still need: {', '.join(missing)}"
                logger.info(f"Returning partial completion response: {response}")
                return response
            else:
                response = "🎉 Perfect! All information collected. Profile complete!"
                logger.info(f"Returning completion response: {response}")
                return response
                
        except Exception as e:
            logger.error(f"Profile update failed: {str(e)}", exc_info=True)
            return f"❌ Error storing data: {str(e)}"

    callback_context._invocation_context.agent.tools = [update_profile]

def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest):
    """Manage the conversation flow based on what's been collected."""
    logger.info("Starting before model callback")
    
    # Get profile data from session state
    collected = callback_context._invocation_context.session.state.get("user_profile_partial", {})

    # Check what's missing and guide the conversation
    missing_fields = [f for f in InvestorProfile.model_fields.keys() if f not in collected]
    
    if missing_fields:
        logger.info(f"Still need to collect: {missing_fields}")
        return  # Continue conversation
    else:
        logger.info("All profile data collected, ready to finalize")
        return

def after_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    """Process the collected profile but don't end the conversation - let orchestrator continue."""
    logger.info("Starting after agent callback")

    # Check for completed profile first, then partial from session state
    user_data = callback_context._invocation_context.session.state.get("user_profile", {})
    if not user_data:
        user_data = callback_context._invocation_context.session.state.get("user_profile_partial", {})

    # Check if we have a complete profile and log it, but don't return content
    try:
        profile = InvestorProfile(**user_data)
        logger.info("Complete profile validated successfully - letting orchestrator continue to next step")
        logger.info(f"Final profile: {json.dumps(profile.model_dump(), indent=2)}")
        return None
    except Exception as e:
        logger.debug(f"Profile not yet complete: {str(e)}")
        return None

def load_instruction():
    with open(PROMPT_FILE) as f:
        return f.read()

class InvestorProfileAgent(LlmAgent):
    _user_input: Optional[InvestorProfile] = PrivateAttr(default=None)

    def __init__(self):
        super().__init__(
            name="investor_profile_agent",
            description="Collects and validates investor profile data.",
            model=os.getenv("MODEL_NAME", "gemini-2.0-flash"),
            instruction=load_instruction(),
            before_agent_callback=before_agent_callback,
            before_model_callback=before_model_callback,
            after_agent_callback=after_agent_callback,
            tools=[], # validate_profile added at runtime
            output_key="profile_collection_result"  # ADK will save agent's response to session state
        )

    async def run_async(self, ctx: CallbackContext) -> AsyncGenerator[Event, None]:
        """Run the profile collection workflow asynchronously."""
        # Just run the parent's run_async, don't inject a synthetic message
        async for event in super().run_async(ctx):
            if hasattr(event, "is_final_response") and event.is_final_response():
                print(f"✅ Complete Investor Profile:\n{event.content.parts[0].text.strip()}")
            yield event