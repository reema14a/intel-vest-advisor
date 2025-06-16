import logging
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.adk.events import Event
from google.genai import types
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, AsyncGenerator, Any
import os
import json
from agents.investor_profile.agent import InvestorProfileAgent
from agents.risk_assessment.agent import RiskAssessmentAgent
from agents.risk_model_selection.agent import RiskModelSelectionAgent
from utils.load_env import load_agent_env
from utils.session_manager import SessionManager
from utils.monitoring import monitoring
from database.session_store import DatabaseSessionStore

# Configure logging
logger = logging.getLogger(__name__)

# Import monitoring conditionally
try:
    from utils.monitoring import monitoring
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    monitoring = None

class ConversationState(BaseModel):
    current_agent: str = Field(default="orchestration")
    collected_data: Dict = Field(default_factory=dict)
    next_steps: List[str] = Field(default_factory=list)
    completed_steps: List[str] = Field(default_factory=list)

class OrchestrationAgent(BaseAgent):
    """Custom orchestration agent for coordinating the investment advisory workflow"""
    
    # --- Field Declarations for Pydantic ---
    # Declare the agents passed during initialization as class attributes with type hints
    investor_profile_agent: InvestorProfileAgent
    risk_assessment_agent: RiskAssessmentAgent
    risk_model_selection_agent: RiskModelSelectionAgent
    db_store: Optional[DatabaseSessionStore] = None
    
    def __init__(self, session_manager: SessionManager):
        """Initialize the orchestration agent
        
        Args:
            session_manager: Session manager instance (for compatibility, but ADK manages state)
        """
        # Load environment variables
        load_agent_env(__file__, "orchestration")
        
        # Create sub-agents as local variables first (ADK pattern)
        investor_profile_agent = InvestorProfileAgent()
        risk_assessment_agent = RiskAssessmentAgent()
        risk_model_selection_agent = RiskModelSelectionAgent()
        
        # Define the sub_agents list for the framework
        sub_agents_list = [
            investor_profile_agent,
            risk_assessment_agent,
            risk_model_selection_agent,
        ]
        
        # Initialize the base agent with individual agents and sub_agents list (now that Pydantic knows about the fields)
        super().__init__(
            name="orchestration_agent",
            description="Orchestrates the investment advisory process through a sequential workflow",
            investor_profile_agent=investor_profile_agent,
            risk_assessment_agent=risk_assessment_agent,
            risk_model_selection_agent=risk_model_selection_agent,
            sub_agents=sub_agents_list,  # Pass the sub_agents list directly
        )
        
        # Store session manager for compatibility (but ADK will manage state)
        self._session_manager = session_manager
        
        # Initialize database session store for reliable persistence
        self.db_store = DatabaseSessionStore()
        
        logger.info("OrchestrationAgent initialized successfully")

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        State-based orchestration logic that determines which agent to run based on current state.
        Maintains state flow within the same conversation turn.
        """
        logger.info(f"[{self.name}] Starting investment advisory workflow")
        
        try:
            # Re-evaluate workflow state dynamically after each step
            while True:
                # Fresh state check on each iteration
                user_profile = ctx.session.state.get("user_profile")
                selected_model = ctx.session.state.get("selected_model")
                risk_profile = ctx.session.state.get("predicted_risk")
                
                logger.info(f"[{self.name}] Current state - Profile: {bool(user_profile and self._is_profile_complete(user_profile))}, Model: {bool(selected_model)}, Risk: {bool(risk_profile)}")
                
                # Step 1: Profile Collection (if not complete)
                if not user_profile or not self._is_profile_complete(user_profile):
                    logger.info(f"[{self.name}] Step 1: Profile collection needed")
                    logger.info(f"[{self.name}] BEFORE calling profile agent - session state: {dict(ctx.session.state)}")
                    
                    # Get session ID for database operations
                    session_id = getattr(ctx.session, 'id', 'unknown_session')
                    
                    # Load current profile from database (reliable persistence)
                    current_partial = self.db_store.get_user_profile_dict(session_id)
                    logger.info(f"[{self.name}] Loaded profile from database: {current_partial}")
                    
                    # Also update session state for compatibility with sub-agents
                    if current_partial:
                        ctx.session.state["user_profile_partial"] = current_partial
                    
                    logger.info(f"[{self.name}] Current partial profile before sub-agent: {current_partial}")
                    
                    # Pass InvocationContext directly - ADK handles CallbackContext conversion internally
                    async for event in self.investor_profile_agent.run_async(ctx):
                        yield event
                    
                    logger.info(f"[{self.name}] AFTER calling profile agent - session state: {dict(ctx.session.state)}")
                    
                    # According to ADK docs, custom agents should manage state directly in session.state
                    # Check if the profile agent collected any data and merge it into our profile
                    profile_result = ctx.session.state.get("profile_collection_result", "")
                    
                    # The profile may have been lost in the event processing - restore it and check for updates
                    # First, check if we have any new profile data from the sub-agent's output
                    updated_partial_profile = ctx.session.state.get("user_profile_partial", {})
                    updated_user_profile = ctx.session.state.get("user_profile", {})
                    
                    # If the partial profile is empty but we had data before, restore it
                    if not updated_partial_profile and current_partial:
                        logger.info(f"[{self.name}] Restoring lost partial profile: {current_partial}")
                        ctx.session.state["user_profile_partial"] = current_partial
                        updated_partial_profile = current_partial
                    
                    logger.info(f"[{self.name}] Profile collection result: {profile_result}")
                    logger.info(f"[{self.name}] Updated user_profile: {updated_user_profile}")
                    logger.info(f"[{self.name}] Updated user_profile_partial: {updated_partial_profile}")
                    
                    # Use database for reliable persistence instead of unreliable session state
                    if updated_partial_profile:
                        # We have some data from this run - merge with what we had before
                        merged_profile = current_partial.copy()
                        merged_profile.update(updated_partial_profile)
                        logger.info(f"[{self.name}] Merging profiles - Previous: {current_partial}, Current: {updated_partial_profile}")
                        logger.info(f"[{self.name}] Merged profile: {merged_profile}")
                        
                        # Save to database for reliable persistence
                        try:
                            self.db_store.create_or_update_user_profile(session_id, merged_profile)
                            logger.info(f"[{self.name}] Saved merged profile to database: {merged_profile}")
                        except Exception as e:
                            logger.error(f"[{self.name}] Failed to save profile to database: {str(e)}")
                        
                        # Also update session state for compatibility
                        ctx.session.state["user_profile_partial"] = merged_profile
                        updated_partial_profile = merged_profile
                    elif current_partial:
                        # No new data, but we had data before - make sure it's in session state
                        logger.info(f"[{self.name}] No new data collected, using existing profile: {current_partial}")
                        ctx.session.state["user_profile_partial"] = current_partial
                        updated_partial_profile = current_partial
                    
                    # Check if profile is complete (either in user_profile or user_profile_partial)
                    profile_complete = False
                    if updated_user_profile and self._is_profile_complete(updated_user_profile):
                        profile_complete = True
                        logger.info(f"[{self.name}] Profile completed (from user_profile), continuing to next step")
                    elif updated_partial_profile and self._is_profile_complete(updated_partial_profile):
                        # Partial profile is complete - promote it to user_profile
                        ctx.session.state["user_profile"] = updated_partial_profile
                        profile_complete = True
                        logger.info(f"[{self.name}] Profile completed (from partial), promoted to user_profile, continuing to next step")
                    
                    if profile_complete:
                        continue  # Re-evaluate workflow state to move to next step
                    else:
                        # If we have partial data, that's progress - continue collecting
                        if updated_partial_profile and len(updated_partial_profile) > len(current_partial):
                            logger.info(f"[{self.name}] Profile collection made progress, continuing conversation")
                        else:
                            logger.info(f"[{self.name}] Profile collection needs more input")
                        return  # Exit and wait for more user input
                
                # Step 2: Model Selection (if not done and profile is complete)
                elif not selected_model:
                    logger.info(f"[{self.name}] Step 2: Model selection needed")
                    # Pass InvocationContext directly - ADK handles CallbackContext conversion internally
                    async for event in self.risk_model_selection_agent.run_async(ctx):
                        yield event
                    
                    # Check if model selection completed
                    updated_selected_model = ctx.session.state.get("selected_model")
                    
                    if updated_selected_model:
                        logger.info(f"[{self.name}] Model selection completed with model: {updated_selected_model}")
                        continue  # Re-evaluate workflow state
                    else:
                        logger.info(f"[{self.name}] Model selection incomplete, exiting")
                        return
                
                # Step 3: Risk Assessment (if not done and previous steps complete)
                elif not risk_profile:
                    logger.info(f"[{self.name}] Step 3: Risk assessment needed")
                    # Pass InvocationContext directly - ADK handles CallbackContext conversion internally
                    async for event in self.risk_assessment_agent.run_async(ctx):
                        yield event
                    
                    # Check if risk assessment completed
                    updated_risk_profile = ctx.session.state.get("predicted_risk")
                    if updated_risk_profile:
                        logger.info(f"[{self.name}] Risk assessment completed, continuing to final step")
                        continue  # Re-evaluate workflow state for final recommendations
                    else:
                        logger.info(f"[{self.name}] Risk assessment incomplete, exiting")
                        return
                
                # Step 4: Generate Final Recommendations (all steps complete)
                else:
                    logger.info(f"[{self.name}] Step 4: All steps complete, generating final recommendations")
                    model_selection_result = ctx.session.state.get("validated_result", {"selectedModel": selected_model, "reason": "Model selected"})
                    recommendations = self._generate_final_recommendations(risk_profile, model_selection_result)
                    
                    if MONITORING_AVAILABLE and monitoring:
                        monitoring.log_agent_interaction(
                            "orchestration",
                            "workflow_complete",
                            {
                                "user_profile": user_profile,
                                "risk_profile": risk_profile,
                                "selected_model": selected_model
                            }
                        )
                    
                    # Create final response event
                    yield Event(
                        author="orchestration_agent",
                        content=types.Content(
                            role="assistant",
                            parts=[types.Part(text=recommendations)]
                        )
                    )
                    
                    # Workflow complete, exit the loop
                    break
            
        except Exception as e:
            logger.error(f"[{self.name}] Error in orchestration workflow: {str(e)}", exc_info=True)
            if MONITORING_AVAILABLE and monitoring:
                monitoring.log_error("orchestration_workflow_error", str(e))
            
            yield Event(
                author="orchestration_agent",
                content=types.Content(
                    role="assistant",
                    parts=[types.Part(text="I apologize, but an error occurred during the advisory process. Please try again.")]
                )
            )

    def _is_profile_complete(self, user_profile: dict) -> bool:
        """Check if the user profile contains all required fields."""
        required_fields = ["age", "education", "income", "emergency_savings", "retirement_planning", "financial_literacy_score"]
        return user_profile and all(field in user_profile for field in required_fields)

    def _generate_final_recommendations(self, risk_profile: dict, model_selection_result: dict) -> str:
        """Generate comprehensive investment recommendations"""
        logger.info("Generating final investment recommendations")
        
        try:
            # Extract key information
            risk_level = risk_profile.get("risk_profile", "Unknown")
            confidence = risk_profile.get("probability", 0)
            model_name = model_selection_result.get("selectedModel", "Unknown")
            model_reason = model_selection_result.get("reason", "No reason provided")
            
            recommendations = f"""
# 🎯 Your Personalized Investment Recommendations

## Risk Profile Analysis
- **Risk Level**: {risk_level}
- **Confidence**: {confidence:.1%}

## Recommended Investment Model
- **Selected Model**: {model_name}
- **Selection Rationale**: {model_reason}

## Investment Strategy Recommendations

### Based on your {risk_level.lower()} risk profile:

#### Portfolio Allocation
- **Stocks**: {self._get_stock_allocation(risk_level)}
- **Bonds**: {self._get_bond_allocation(risk_level)}
- **Alternative Investments**: {self._get_alternative_allocation(risk_level)}

#### Recommended Actions
{self._get_recommended_actions(risk_level)}

#### Risk Management
{self._get_risk_management_advice(risk_level)}

---
*This analysis is based on your responses and our proprietary risk assessment model. Please consult with a financial advisor for personalized advice.*
"""
            return recommendations.strip()
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return "I apologize, but I encountered an error while generating your recommendations. Please try again."
    
    def _get_stock_allocation(self, risk_level: str) -> str:
        allocations = {
            "conservative": "30-40%",
            "moderate": "50-60%", 
            "aggressive": "70-80%"
        }
        return allocations.get(risk_level.lower(), "50-60%")
    
    def _get_bond_allocation(self, risk_level: str) -> str:
        allocations = {
            "conservative": "50-60%",
            "moderate": "30-40%",
            "aggressive": "10-20%"
        }
        return allocations.get(risk_level.lower(), "30-40%")
    
    def _get_alternative_allocation(self, risk_level: str) -> str:
        allocations = {
            "conservative": "5-10%",
            "moderate": "10-15%",
            "aggressive": "15-20%"
        }
        return allocations.get(risk_level.lower(), "10-15%")
    
    def _get_recommended_actions(self, risk_level: str) -> str:
        actions = {
            "conservative": """
- Focus on dividend-paying stocks and high-grade bonds
- Consider index funds for diversification
- Maintain 6-12 months emergency fund
- Regular portfolio rebalancing every 6 months
""",
            "moderate": """
- Balanced mix of growth and value stocks
- Include international diversification
- Consider dollar-cost averaging for investments
- Review and rebalance portfolio quarterly
""",
            "aggressive": """
- Focus on growth stocks and emerging markets
- Consider sector-specific ETFs
- Higher allocation to individual stocks
- Monitor portfolio monthly and adjust as needed
"""
        }
        return actions.get(risk_level.lower(), actions["moderate"])
    
    def _get_risk_management_advice(self, risk_level: str) -> str:
        advice = {
            "conservative": """
- Prioritize capital preservation over growth
- Avoid high-volatility investments
- Consider Treasury Inflation-Protected Securities (TIPS)
- Regular income through dividends and bond coupons
""",
            "moderate": """
- Balance growth potential with downside protection
- Use stop-loss orders for individual stock positions
- Diversify across multiple asset classes
- Regular review of investment performance
""",
            "aggressive": """
- Accept higher volatility for potential higher returns
- Use diversification to manage concentration risk
- Consider hedging strategies during market uncertainty
- Stay informed about market trends and economic indicators
"""
        }
        return advice.get(risk_level.lower(), advice["moderate"]) 