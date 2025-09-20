"""
Configuration settings for the Tele-Radiology Backend
"""
import os
from pathlib import Path

try:
    from pydantic import BaseSettings
except ImportError:
    # Fallback for when pydantic is not installed
    class BaseSettings:
        def __init__(self):
            pass
        
        class Config:
            env_file = ".env"
            case_sensitive = False


class Settings(BaseSettings):
    # Application settings
    app_name: str = "Tele-Radiology Backend"
    debug: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent
    storage_dir: Path = base_dir / "storage"
    original_images_dir: Path = storage_dir / "original"
    annotated_images_dir: Path = storage_dir / "annotated"
    
    # Database
    database_url: str = f"sqlite:///{base_dir}/radiology.db"
    
    # Model settings
    max_image_size: int = 2048  # Maximum image dimension in pixels
    supported_formats: list = [".jpg", ".jpeg", ".png", ".dcm", ".dicom"]
    
    # Inference settings
    confidence_threshold: float = 0.5  # Medical-grade threshold: 50%
    max_predictions: int = 5  # Realistic number of findings
    
    # Performance settings
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    inference_timeout: int = 30  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Create necessary directories
settings.storage_dir.mkdir(exist_ok=True)
settings.original_images_dir.mkdir(exist_ok=True)
settings.annotated_images_dir.mkdir(exist_ok=True)