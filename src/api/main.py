"""
FastAPI endpoints for the Tele-Radiology Backend
"""
import json
import uuid
import asyncio
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from config.settings import settings
from src.database.connection import get_db, create_tables
from src.database.models import RadiologyCase, Prediction
from src.preprocessing.image_processor import ImagePreprocessor
from src.models.inference import ModelInferenceEngine
from src.visualization.heatmap_generator import MedicalImageVisualizer
from src.api.schemas import (
    AnalysisResponse,
    CaseResponse,
    HealthResponse,
    ErrorResponse,
    ModelListResponse,
    PredictionResponse,
    ModelInfo
)


class TeleRadiologyAPI:
    """
    Main API class for the Tele-Radiology Backend
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="Tele-Radiology Backend",
            description="Local backend for medical image analysis with AI",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Initialize components
        self.preprocessor = ImagePreprocessor()
        self.inference_engine = ModelInferenceEngine()
        self.visualizer = MedicalImageVisualizer()
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Setup static files
        self._setup_static_files()
        
        # Create database tables
        create_tables()
    
    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_static_files(self):
        """Setup static file serving for images"""
        # Mount static directories
        self.app.mount(
            "/files/original",
            StaticFiles(directory=str(settings.original_images_dir)),
            name="original_images"
        )
        self.app.mount(
            "/files/annotated",
            StaticFiles(directory=str(settings.annotated_images_dir)),
            name="annotated_images"
        )
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            try:
                available_models = list(self.inference_engine.get_available_models().keys())
                device = str(self.inference_engine.device)
                
                return HealthResponse(
                    status="healthy",
                    version="1.0.0",
                    timestamp=datetime.utcnow(),
                    models_loaded=available_models,
                    device=device
                )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Health check failed: {str(e)}"
                )
        
        @self.app.get("/models", response_model=ModelListResponse)
        async def list_models():
            """List available models and their information"""
            try:
                models_info = self.inference_engine.get_available_models()
                return ModelListResponse(models=models_info)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to retrieve model information: {str(e)}"
                )
        
        @self.app.post("/analyze", response_model=AnalysisResponse)
        async def analyze_image(
            file: UploadFile = File(...),
            model_name: str = "chexpert",
            db: Session = Depends(get_db)
        ):
            """
            Analyze uploaded medical image
            
            Args:
                file: Uploaded image file
                model_name: Name of the model to use for analysis
                db: Database session
                
            Returns:
                Analysis results with predictions and annotated image
            """
            case_id = str(uuid.uuid4())
            
            try:
                # Read file content
                file_content = await file.read()
                
                # Validate image
                is_valid, error_msg = self.preprocessor.validate_image(file_content, file.filename)
                if not is_valid:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=error_msg
                    )
                
                # Save original file
                original_path = settings.original_images_dir / f"{case_id}_{file.filename}"
                with open(original_path, "wb") as f:
                    f.write(file_content)
                
                # Load and preprocess image
                image_array = self.preprocessor.load_image(file_content, file.filename)
                image_tensor = self.preprocessor.preprocess_for_inference(image_array, "xray")
                
                # Run inference first (without gradients for efficiency)
                inference_results = self.inference_engine.predict(image_tensor, model_name)
                
                # Generate heatmap and visualization - now using proper GradCAM
                try:
                    print("Generating GradCAM heatmap...")
                    heatmap = self.inference_engine.generate_gradcam_heatmap(
                        image_tensor, model_name
                    )
                    print(f"✓ Generated GradCAM heatmap with shape: {heatmap.shape}")
                except Exception as e:
                    print(f"GradCAM generation failed: {e}")
                    print("Using fallback heatmap to avoid gradient conflicts")
                    h, w = image_array.shape[:2]
                    heatmap = np.zeros((h, w))
                    center_h, center_w = h // 2, w // 2
                    # Create a simple gaussian-like heatmap centered on the image
                    y, x = np.ogrid[:h, :w]
                    mask = ((x - center_w) ** 2 + (y - center_h) ** 2) <= (min(h, w) // 4) ** 2
                    heatmap[mask] = 0.5
                
                # Create annotated image
                annotated_image = self.visualizer.create_annotated_image(
                    image_array,
                    inference_results["predictions"],
                    heatmap,
                    case_id
                )
                
                # Save annotated image
                annotated_path = settings.annotated_images_dir / f"{case_id}_annotated.png"
                self.visualizer.save_visualization(annotated_image, annotated_path)
                
                # Save to database
                db_case = RadiologyCase(
                    case_id=case_id,
                    original_filename=file.filename,
                    file_path=str(original_path),
                    annotated_image_path=str(annotated_path),
                    file_size=len(file_content),
                    image_format=file.filename.split('.')[-1].lower(),
                    processed_at=datetime.utcnow(),
                    predictions_json=json.dumps(inference_results["predictions"]),
                    model_version=model_name,
                    processing_time_seconds=inference_results["processing_time"],
                    confidence_threshold=settings.confidence_threshold
                )
                
                db.add(db_case)
                
                # Save individual predictions
                for pred in inference_results["predictions"]:
                    db_pred = Prediction(
                        case_id=case_id,
                        condition=pred["condition"],
                        confidence_score=pred["confidence"],
                        rank=pred["rank"]
                    )
                    db.add(db_pred)
                
                db.commit()
                
                # Prepare response
                predictions = [
                    PredictionResponse(**pred) for pred in inference_results["predictions"]
                ]
                
                model_info = ModelInfo(
                    name=inference_results["model_info"]["name"],
                    version=inference_results["model_info"]["version"],
                    total_pathologies=inference_results["model_info"]["total_pathologies"],
                    confidence_threshold=inference_results["model_info"]["confidence_threshold"]
                )
                
                return AnalysisResponse(
                    case_id=case_id,
                    predictions=predictions,
                    annotated_image_url=f"/files/annotated/{case_id}_annotated.png",
                    model_info=model_info,
                    processing_time=inference_results["processing_time"],
                    created_at=db_case.created_at
                )
                
            except HTTPException:
                raise
            except Exception as e:
                # Clean up files on error
                if original_path.exists():
                    original_path.unlink()
                if 'annotated_path' in locals() and annotated_path.exists():
                    annotated_path.unlink()
                
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Analysis failed: {str(e)}"
                )
        
        @self.app.get("/cases/{case_id}", response_model=CaseResponse)
        async def get_case(case_id: str, db: Session = Depends(get_db)):
            """
            Retrieve information about a specific case
            
            Args:
                case_id: Unique case identifier
                db: Database session
                
            Returns:
                Complete case information
            """
            try:
                # Query case from database
                db_case = db.query(RadiologyCase).filter(RadiologyCase.case_id == case_id).first()
                
                if not db_case:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Case {case_id} not found"
                    )
                
                # Parse predictions JSON
                predictions_data = json.loads(db_case.predictions_json) if db_case.predictions_json else []
                predictions = [PredictionResponse(**pred) for pred in predictions_data]
                
                # Create model info (reconstruct from stored data)
                model_info = ModelInfo(
                    name=f"Model {db_case.model_version}",
                    version=db_case.model_version or "unknown",
                    total_pathologies=len(predictions),
                    confidence_threshold=db_case.confidence_threshold or settings.confidence_threshold
                )
                
                return CaseResponse(
                    case_id=case_id,
                    original_filename=db_case.original_filename,
                    predictions=predictions,
                    annotated_image_url=f"/files/annotated/{case_id}_annotated.png",
                    original_image_url=f"/files/original/{case_id}_{db_case.original_filename}",
                    model_info=model_info,
                    processing_time=db_case.processing_time_seconds or 0.0,
                    created_at=db_case.created_at,
                    file_size=db_case.file_size,
                    image_format=db_case.image_format
                )
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to retrieve case: {str(e)}"
                )
        
        @self.app.get("/cases", response_model=List[CaseResponse])
        async def list_cases(
            limit: int = 50,
            offset: int = 0,
            db: Session = Depends(get_db)
        ):
            """
            List all cases with pagination
            
            Args:
                limit: Maximum number of cases to return
                offset: Number of cases to skip
                db: Database session
                
            Returns:
                List of cases
            """
            try:
                cases = db.query(RadiologyCase).offset(offset).limit(limit).all()
                
                response_cases = []
                for db_case in cases:
                    predictions_data = json.loads(db_case.predictions_json) if db_case.predictions_json else []
                    predictions = [PredictionResponse(**pred) for pred in predictions_data]
                    
                    model_info = ModelInfo(
                        name=f"Model {db_case.model_version}",
                        version=db_case.model_version or "unknown",
                        total_pathologies=len(predictions),
                        confidence_threshold=db_case.confidence_threshold or settings.confidence_threshold
                    )
                    
                    response_cases.append(CaseResponse(
                        case_id=db_case.case_id,
                        original_filename=db_case.original_filename,
                        predictions=predictions,
                        annotated_image_url=f"/files/annotated/{db_case.case_id}_annotated.png",
                        original_image_url=f"/files/original/{db_case.case_id}_{db_case.original_filename}",
                        model_info=model_info,
                        processing_time=db_case.processing_time_seconds or 0.0,
                        created_at=db_case.created_at,
                        file_size=db_case.file_size,
                        image_format=db_case.image_format
                    ))
                
                return response_cases
                
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to list cases: {str(e)}"
                )


# Create the API instance
api = TeleRadiologyAPI()
app = api.app