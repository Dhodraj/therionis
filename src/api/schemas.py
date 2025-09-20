"""
Pydantic models for API requests and responses
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class PredictionResponse(BaseModel):
    """Response model for individual predictions"""
    condition: str = Field(..., description="Name of the detected condition")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    rank: int = Field(..., ge=1, description="Ranking by confidence")


class ModelInfo(BaseModel):
    """Model information"""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    total_pathologies: int = Field(..., description="Total number of pathologies the model can detect")
    confidence_threshold: float = Field(..., description="Confidence threshold used")


class AnalysisResponse(BaseModel):
    """Response model for image analysis"""
    case_id: str = Field(..., description="Unique case identifier")
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    annotated_image_url: str = Field(..., description="URL to access the annotated image")
    model_info: ModelInfo = Field(..., description="Information about the model used")
    processing_time: float = Field(..., description="Processing time in seconds")
    created_at: datetime = Field(..., description="Timestamp when analysis was performed")


class CaseResponse(BaseModel):
    """Response model for retrieving a specific case"""
    case_id: str = Field(..., description="Unique case identifier")
    original_filename: str = Field(..., description="Original filename of uploaded image")
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    annotated_image_url: str = Field(..., description="URL to access the annotated image")
    original_image_url: str = Field(..., description="URL to access the original image")
    model_info: ModelInfo = Field(..., description="Information about the model used")
    processing_time: float = Field(..., description="Processing time in seconds")
    created_at: datetime = Field(..., description="Timestamp when analysis was performed")
    file_size: int = Field(..., description="Size of the original file in bytes")
    image_format: str = Field(..., description="Format of the image")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current timestamp")
    models_loaded: List[str] = Field(..., description="List of loaded models")
    device: str = Field(..., description="Device being used (CPU/GPU)")


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(..., description="Error timestamp")


class ModelListResponse(BaseModel):
    """Response model for listing available models"""
    models: Dict[str, Dict[str, Any]] = Field(..., description="Available models and their information")