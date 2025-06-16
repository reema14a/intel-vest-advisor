"""
Database models for session and profile data persistence.
"""

from sqlalchemy import Column, String, DateTime, Text, Integer, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator
import json
from datetime import datetime

Base = declarative_base()


class JSONField(TypeDecorator):
    """Custom SQLAlchemy type for storing JSON data."""
    impl = Text
    
    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None
    
    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return None


class SessionData(Base):
    """Store ADK session data and state."""
    __tablename__ = 'session_data'
    
    id = Column(String, primary_key=True)  # ADK session ID
    user_id = Column(String, nullable=False, index=True)
    ui_session_id = Column(String, nullable=False, index=True)
    app_name = Column(String, nullable=False)
    session_state = Column(JSONField, default=dict)  # Store session.state as JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to user profiles
    user_profiles = relationship("UserProfile", back_populates="session", cascade="all, delete-orphan")


class UserProfile(Base):
    """Store user investment profile data."""
    __tablename__ = 'user_profiles'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey('session_data.id'), nullable=False)
    
    # Profile completion status
    is_complete = Column(Boolean, default=False)
    
    # Investment profile fields
    age = Column(Integer, nullable=True)  # 1-6 age groups
    education = Column(Integer, nullable=True)  # 1-7 education levels
    income = Column(Integer, nullable=True)  # 1-10 income brackets
    emergency_savings = Column(Integer, nullable=True)  # 1-2 yes/no
    retirement_planning = Column(Integer, nullable=True)  # 1-3 yes/no/unknown
    financial_literacy_score = Column(Integer, nullable=True)  # 0-3 score
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship back to session
    session = relationship("SessionData", back_populates="user_profiles")
    
    def to_dict(self):
        """Convert profile to dictionary format expected by agents."""
        profile_dict = {}
        
        if self.age is not None:
            profile_dict['age'] = self.age
        if self.education is not None:
            profile_dict['education'] = self.education
        if self.income is not None:
            profile_dict['income'] = self.income
        if self.emergency_savings is not None:
            profile_dict['emergency_savings'] = self.emergency_savings
        if self.retirement_planning is not None:
            profile_dict['retirement_planning'] = self.retirement_planning
        if self.financial_literacy_score is not None:
            profile_dict['financial_literacy_score'] = self.financial_literacy_score
            
        return profile_dict
    
    def update_from_dict(self, data):
        """Update profile from dictionary data."""
        if 'age' in data:
            self.age = data['age']
        if 'education' in data:
            self.education = data['education']
        if 'income' in data:
            self.income = data['income']
        if 'emergency_savings' in data:
            self.emergency_savings = data['emergency_savings']
        if 'retirement_planning' in data:
            self.retirement_planning = data['retirement_planning']
        if 'financial_literacy_score' in data:
            self.financial_literacy_score = data['financial_literacy_score']
        
        # Check if profile is complete
        required_fields = ['age', 'education', 'income', 'emergency_savings', 'retirement_planning', 'financial_literacy_score']
        self.is_complete = all(getattr(self, field) is not None for field in required_fields)
        
        self.updated_at = datetime.utcnow()


class RiskAssessment(Base):
    """Store risk assessment results."""
    __tablename__ = 'risk_assessments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey('session_data.id'), nullable=False)
    
    # Risk assessment results
    risk_score = Column(Integer, nullable=True)
    risk_category = Column(String, nullable=True)  # conservative, moderate, aggressive
    assessment_data = Column(JSONField, default=dict)  # Store full assessment results
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelSelection(Base):
    """Store selected investment model data."""
    __tablename__ = 'model_selections'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey('session_data.id'), nullable=False)
    
    # Model selection data
    selected_model = Column(String, nullable=True)
    model_parameters = Column(JSONField, default=dict)
    selection_rationale = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow) 