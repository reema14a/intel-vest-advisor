"""
Database service for session and profile data persistence.
"""

import logging
from typing import Dict, Optional, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timedelta
import os

from .models import Base, SessionData, UserProfile, RiskAssessment, ModelSelection

logger = logging.getLogger(__name__)


class DatabaseSessionStore:
    """Database-backed session store for reliable persistence."""
    
    def __init__(self, database_url: str = None):
        """Initialize the database session store."""
        if database_url is None:
            # Default to SQLite database in project root
            database_url = f"sqlite:///session_data.db"
        
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(bind=self.engine)
        logger.info(f"Database session store initialized with URL: {database_url}")
    
    def get_db_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def create_or_update_session(self, session_id: str, user_id: str, ui_session_id: str, 
                                app_name: str, session_state: Dict[str, Any] = None) -> SessionData:
        """Create or update a session in the database."""
        if session_state is None:
            session_state = {}
        
        with self.get_db_session() as db:
            try:
                # Try to get existing session
                session_data = db.query(SessionData).filter(SessionData.id == session_id).first()
                
                if session_data:
                    # Update existing session
                    session_data.session_state = session_state
                    session_data.updated_at = datetime.utcnow()
                    logger.info(f"Updated existing session {session_id}")
                else:
                    # Create new session
                    session_data = SessionData(
                        id=session_id,
                        user_id=user_id,
                        ui_session_id=ui_session_id,
                        app_name=app_name,
                        session_state=session_state
                    )
                    db.add(session_data)
                    logger.info(f"Created new session {session_id}")
                
                db.commit()
                db.refresh(session_data)
                return session_data
                
            except SQLAlchemyError as e:
                db.rollback()
                logger.error(f"Database error creating/updating session {session_id}: {str(e)}")
                raise
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get a session from the database."""
        with self.get_db_session() as db:
            try:
                session_data = db.query(SessionData).filter(SessionData.id == session_id).first()
                if session_data:
                    logger.debug(f"Retrieved session {session_id} with state: {session_data.session_state}")
                return session_data
            except SQLAlchemyError as e:
                logger.error(f"Database error retrieving session {session_id}: {str(e)}")
                return None
    
    def update_session_state(self, session_id: str, session_state: Dict[str, Any]) -> bool:
        """Update session state in the database."""
        with self.get_db_session() as db:
            try:
                session_data = db.query(SessionData).filter(SessionData.id == session_id).first()
                if session_data:
                    session_data.session_state = session_state
                    session_data.updated_at = datetime.utcnow()
                    db.commit()
                    logger.debug(f"Updated session state for {session_id}: {session_state}")
                    return True
                else:
                    logger.warning(f"Session {session_id} not found for state update")
                    return False
            except SQLAlchemyError as e:
                db.rollback()
                logger.error(f"Database error updating session state {session_id}: {str(e)}")
                return False
    
    def get_user_profile(self, session_id: str) -> Optional[UserProfile]:
        """Get user profile for a session."""
        with self.get_db_session() as db:
            try:
                profile = db.query(UserProfile).filter(UserProfile.session_id == session_id).first()
                return profile
            except SQLAlchemyError as e:
                logger.error(f"Database error retrieving user profile for {session_id}: {str(e)}")
                return None
    
    def create_or_update_user_profile(self, session_id: str, profile_data: Dict[str, Any]) -> UserProfile:
        """Create or update user profile."""
        with self.get_db_session() as db:
            try:
                # Try to get existing profile
                profile = db.query(UserProfile).filter(UserProfile.session_id == session_id).first()
                
                if profile:
                    # Update existing profile
                    profile.update_from_dict(profile_data)
                    logger.info(f"Updated user profile for session {session_id}: {profile_data}")
                else:
                    # Create new profile
                    profile = UserProfile(session_id=session_id)
                    profile.update_from_dict(profile_data)
                    db.add(profile)
                    logger.info(f"Created new user profile for session {session_id}: {profile_data}")
                
                db.commit()
                db.refresh(profile)
                return profile
                
            except SQLAlchemyError as e:
                db.rollback()
                logger.error(f"Database error creating/updating user profile for {session_id}: {str(e)}")
                raise
    
    def get_user_profile_dict(self, session_id: str) -> Dict[str, Any]:
        """Get user profile as dictionary."""
        profile = self.get_user_profile(session_id)
        if profile:
            return profile.to_dict()
        return {}
    
    def is_profile_complete(self, session_id: str) -> bool:
        """Check if user profile is complete."""
        profile = self.get_user_profile(session_id)
        return profile.is_complete if profile else False
    
    def save_risk_assessment(self, session_id: str, risk_score: int, risk_category: str, 
                           assessment_data: Dict[str, Any] = None) -> RiskAssessment:
        """Save risk assessment results."""
        if assessment_data is None:
            assessment_data = {}
        
        with self.get_db_session() as db:
            try:
                # Create or update risk assessment
                assessment = db.query(RiskAssessment).filter(RiskAssessment.session_id == session_id).first()
                
                if assessment:
                    assessment.risk_score = risk_score
                    assessment.risk_category = risk_category
                    assessment.assessment_data = assessment_data
                    assessment.updated_at = datetime.utcnow()
                else:
                    assessment = RiskAssessment(
                        session_id=session_id,
                        risk_score=risk_score,
                        risk_category=risk_category,
                        assessment_data=assessment_data
                    )
                    db.add(assessment)
                
                db.commit()
                db.refresh(assessment)
                logger.info(f"Saved risk assessment for session {session_id}: {risk_category} (score: {risk_score})")
                return assessment
                
            except SQLAlchemyError as e:
                db.rollback()
                logger.error(f"Database error saving risk assessment for {session_id}: {str(e)}")
                raise
    
    def save_model_selection(self, session_id: str, selected_model: str, 
                           model_parameters: Dict[str, Any] = None, 
                           selection_rationale: str = None) -> ModelSelection:
        """Save model selection results."""
        if model_parameters is None:
            model_parameters = {}
        
        with self.get_db_session() as db:
            try:
                # Create or update model selection
                selection = db.query(ModelSelection).filter(ModelSelection.session_id == session_id).first()
                
                if selection:
                    selection.selected_model = selected_model
                    selection.model_parameters = model_parameters
                    selection.selection_rationale = selection_rationale
                    selection.updated_at = datetime.utcnow()
                else:
                    selection = ModelSelection(
                        session_id=session_id,
                        selected_model=selected_model,
                        model_parameters=model_parameters,
                        selection_rationale=selection_rationale
                    )
                    db.add(selection)
                
                db.commit()
                db.refresh(selection)
                logger.info(f"Saved model selection for session {session_id}: {selected_model}")
                return selection
                
            except SQLAlchemyError as e:
                db.rollback()
                logger.error(f"Database error saving model selection for {session_id}: {str(e)}")
                raise
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up sessions older than specified days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        with self.get_db_session() as db:
            try:
                # Delete old sessions (cascades to related data)
                deleted_count = db.query(SessionData).filter(
                    SessionData.updated_at < cutoff_date
                ).delete()
                
                db.commit()
                logger.info(f"Cleaned up {deleted_count} sessions older than {days_old} days")
                return deleted_count
                
            except SQLAlchemyError as e:
                db.rollback()
                logger.error(f"Database error during cleanup: {str(e)}")
                return 0 