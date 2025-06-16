import gradio as gr
import logging
from typing import Dict, List, Tuple
from datetime import datetime
from utils.session_manager import SessionManager
from agents.orchestration.agent import OrchestrationAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
import asyncio
import uuid

logger = logging.getLogger(__name__)

class InvestmentAdvisorUI:
    """Handles the Gradio UI interface for the Investment Advisor."""
    
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.agent = OrchestrationAgent(session_manager)
        
        # Initialize ADK session service and runner
        self.session_service = InMemorySessionService()
        self.app_name = "investment_advisor_app"
        self.runner = Runner(
            agent=self.agent,
            session_service=self.session_service,
            app_name=self.app_name
        )
        
        # Track ADK sessions by user session ID
        self.adk_sessions = {}
        
        logger.info("InvestmentAdvisorUI initialized")
    
    def _format_message(self, role: str, content: str) -> str:
        """Format message with timestamp and role."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"[{timestamp}] **{role.title()}**: {content}"
    
    def _get_profile_display(self, session, adk_session=None) -> Dict:
        """Format session data for display."""
        if not session:
            return {}
        
        display_data = {
            "User ID": session.user_id,
            "Name": session.name,
            "Email": session.email,
            "Session ID": session.session_id,
            "Created": session.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "Last Updated": session.updated_at.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add investment profile data if available
        if adk_session and hasattr(adk_session, 'state'):
            user_profile = adk_session.state.get("user_profile", {})
            if user_profile:
                display_data["Investment Profile"] = {
                    "Age Group": user_profile.get("age", "Not collected"),
                    "Education": user_profile.get("education", "Not collected"), 
                    "Income": user_profile.get("income", "Not collected"),
                    "Emergency Savings": user_profile.get("emergency_savings", "Not collected"),
                    "Retirement Planning": user_profile.get("retirement_planning", "Not collected"),
                    "Financial Literacy Score": user_profile.get("financial_literacy_score", "Not collected")
                }
        
        return display_data
    
    async def _process_message(self, message: str, history: List, session) -> Tuple[List, str, Dict]:
        """Process a message through the orchestrator agent."""
        if not session:
            return history, "", {"error": "No active session"}
        
        # Add user message to history (messages format)
        history.append({"role": "user", "content": message})
        
        try:
            # Get or create ADK session for this user session
            # Use a consistent session_id based on our UI session
            adk_session_id = f"adk_{session.session_id}"
            
            if session.session_id not in self.adk_sessions:
                try:
                    # Try to get existing session first
                    adk_session = await self.session_service.get_session(
                        app_name=self.app_name,
                        user_id=session.user_id,
                        session_id=adk_session_id
                    )
                    if adk_session:
                        logger.info(f"♻️ Retrieved existing ADK session: {adk_session_id} with state: {adk_session.state}")
                    else:
                        # Create new session only if it doesn't exist
                        adk_session = await self.session_service.create_session(
                            app_name=self.app_name,
                            user_id=session.user_id,
                            session_id=adk_session_id,
                            state={}  # Start with empty state only for truly new sessions
                        )
                        logger.info(f"🆕 Created new ADK session: {adk_session_id} for UI session: {session.session_id}")
                except Exception as e:
                    # If get_session fails, create a new one
                    adk_session = await self.session_service.create_session(
                        app_name=self.app_name,
                        user_id=session.user_id,
                        session_id=adk_session_id,
                        state={}
                    )
                    logger.info(f"🆕 Created new ADK session after get failed: {adk_session_id}")
                
                self.adk_sessions[session.session_id] = adk_session
            else:
                adk_session = self.adk_sessions[session.session_id]
                logger.info(f"♻️ Reusing cached ADK session: {adk_session.id} for UI session: {session.session_id}")
                logger.info(f"📊 Cached ADK Session state: {adk_session.state}")
                
                # Also try to get the session from session service to compare
                try:
                    fresh_session = await self.session_service.get_session(
                        app_name=self.app_name,
                        user_id=session.user_id,
                        session_id=adk_session_id
                    )
                    if fresh_session:
                        logger.info(f"📊 Fresh ADK Session state from service: {fresh_session.state}")
                        if fresh_session.state != adk_session.state:
                            logger.warning(f"⚠️ Session state mismatch! Cached: {adk_session.state}, Fresh: {fresh_session.state}")
                            # Use the fresh session
                            adk_session = fresh_session
                            self.adk_sessions[session.session_id] = fresh_session
                            logger.info(f"🔄 Updated cached session with fresh state")
                    else:
                        logger.warning(f"⚠️ Could not retrieve fresh session from service")
                except Exception as e:
                    logger.error(f"❌ Error getting fresh session: {str(e)}")
            
            # Create user content using ADK types
            user_content = types.Content(role='user', parts=[types.Part(text=message)])
            
            response_text = ""
            
            # Use ADK runner to process the message as per official documentation
            async for event in self.runner.run_async(
                user_id=session.user_id, 
                session_id=adk_session_id,  # Use the same predictable session ID
                new_message=user_content
            ):
                # Handle streaming content deltas
                if hasattr(event, 'content_part_delta') and event.content_part_delta:
                    if event.content_part_delta.text:
                        response_text += event.content_part_delta.text
                
                # Handle final response
                if event.is_final_response() and event.content:
                    if event.content.parts:
                        for part in event.content.parts:
                            if part.text:
                                response_text += part.text
            
            # After processing, refresh the session from session service to get updated state
            try:
                updated_adk_session = await self.session_service.get_session(
                    app_name=self.app_name,
                    user_id=session.user_id,
                    session_id=adk_session_id
                )
                if updated_adk_session:
                    self.adk_sessions[session.session_id] = updated_adk_session
                    logger.info(f"🔄 Refreshed ADK session state: {updated_adk_session.state}")
                else:
                    logger.warning(f"⚠️ Could not retrieve updated session state for {adk_session_id}")
            except Exception as e:
                logger.error(f"❌ Error refreshing session state: {str(e)}")
            
            # Update history with assistant response (messages format)
            history.append({"role": "assistant", "content": response_text})
            
            return history, "", self._get_profile_display(session, adk_session)
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            error_message = f"I apologize, but I encountered an error: {str(e)}"
            history.append({"role": "assistant", "content": error_message})
            return history, "", {"error": str(e)}
    
    def create_ui(self):
        """Create the Gradio interface."""
        with gr.Blocks(title="Investment Advisor") as demo:
            gr.Markdown("# Investment Advisor")
            
            with gr.Row():
                with gr.Column():
                    name_input = gr.Textbox(label="Name")
                    email_input = gr.Textbox(label="Email")
                    start_btn = gr.Button("Start Session")
                
                with gr.Column():
                    profile_output = gr.JSON(label="User Profile")
            
            with gr.Row():
                
                with gr.Column():
                    msg = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here...",
                        lines=2
                    )
                    submit_btn = gr.Button("Send")
                chatbot = gr.Chatbot(height=600, type='messages')
            
            # Session management
            session_id = gr.State("")
            
            def start_session(name: str, email: str) -> Tuple[str, Dict]:
                """Start a new session."""
                if not name or not email:
                    return "", {"error": "Name and email are required"}
                
                profile = self.session_manager.create_session(name, email)
                return profile.session_id, self._get_profile_display(profile)
            
            def process_message(message: str, history: List, sid: str) -> Tuple[List, str, Dict]:
                """Process user message and update chat history."""
                if not sid:
                    return history, "", {"error": "No active session"}
                
                # Get current session
                session = self.session_manager.get_session(sid)
                if not session:
                    return history, "", {"error": "Session expired"}
                
                # Process message through orchestrator
                return asyncio.run(self._process_message(message, history, session))
            
            # Set up event handlers
            start_btn.click(
                start_session,
                inputs=[name_input, email_input],
                outputs=[session_id, profile_output]
            )
            
            submit_btn.click(
                process_message,
                inputs=[msg, chatbot, session_id],
                outputs=[chatbot, msg, profile_output]
            )
            
            msg.submit(
                process_message,
                inputs=[msg, chatbot, session_id],
                outputs=[chatbot, msg, profile_output]
            )
        
        return demo 