"""
Database models for the Tele-Radiology Backend
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Float, Text, Integer
from sqlalchemy.dialects.postgresql import UUID
from src.database.connection import Base


class RadiologyCase(Base):
    """
    Model for storing radiology cases and their analysis results
    """
    __tablename__ = "radiology_cases"
    
    # Primary key
    case_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # File information
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    annotated_image_path = Column(String, nullable=True)
    file_size = Column(Integer, nullable=False)
    image_format = Column(String, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    
    # Analysis results (stored as JSON string)
    predictions_json = Column(Text, nullable=True)
    
    # Metadata
    model_version = Column(String, nullable=True)
    processing_time_seconds = Column(Float, nullable=True)
    confidence_threshold = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<RadiologyCase(case_id={self.case_id}, filename={self.original_filename})>"


class Prediction(Base):
    """
    Model for storing individual predictions for each case
    """
    __tablename__ = "predictions"
    
    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Foreign key to radiology case
    case_id = Column(String, nullable=False)
    
    # Prediction details
    condition = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)  # Ranking by confidence
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<Prediction(case_id={self.case_id}, condition={self.condition}, confidence={self.confidence_score})>"