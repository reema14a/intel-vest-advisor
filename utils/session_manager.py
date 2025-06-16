import logging
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    user_id: str
    session_id: str
    name: str
    email: str
    state: Dict
    created_at: datetime
    updated_at: datetime

class SessionManager:
    """Manages user sessions and state persistence."""
    
    def __init__(self):
        self._sessions: Dict[str, UserProfile] = {}
        logger.info("SessionManager initialized")
    
    def create_session(self, name: str, email: str) -> UserProfile:
        """Create a new session with user information."""
        user_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        profile = UserProfile(
            user_id=user_id,
            session_id=session_id,
            name=name,
            email=email,
            state={},
            created_at=now,
            updated_at=now
        )
        
        self._sessions[session_id] = profile
        logger.info(f"Created new session for user: {name} ({email})")
        return profile
    
    def get_session(self, session_id: str) -> Optional[UserProfile]:
        """Get session by ID."""
        return self._sessions.get(session_id)
    
    def update_session_state(self, session_id: str, state_updates: Dict) -> Optional[UserProfile]:
        """Update session state with new values."""
        session = self.get_session(session_id)
        if not session:
            logger.error(f"Session not found: {session_id}")
            return None
            
        # Update state
        session.state.update(state_updates)
        session.updated_at = datetime.now()
        
        logger.debug(f"Updated session state for {session_id}")
        return session
    
    def get_session_state(self, session_id: str) -> Optional[Dict]:
        """Get current session state."""
        session = self.get_session(session_id)
        return session.state if session else None
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Cleared session: {session_id}")
            return True
        return False 