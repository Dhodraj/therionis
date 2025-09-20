# Tele-Radiology Backend MVP

A local backend server for medical image analysis using pretrained deep learning models. This system processes X-rays, CT scans, and MRI images to provide automated pathology detection with confidence scores and visual annotations.

## 🚀 Features

- **Medical Image Analysis**: Supports X-ray, CT, and MRI image analysis
- **AI-Powered Detection**: Uses pretrained models (TorchXRayVision) for pathology detection
- **Visual Annotations**: Generates heatmaps and annotated images with predictions
- **REST API**: FastAPI-based endpoints for easy integration
- **Local Storage**: SQLite database and filesystem storage for complete offline operation
- **Real-time Processing**: Fast inference with < 5s processing time per image

## 📋 Requirements

- Python 3.10+
- PyTorch
- FastAPI
- Medical imaging libraries (TorchXRayVision, MONAI)
- OpenCV for image processing

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd classifier
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env file as needed
   ```

## 🚀 Quick Start

1. **Start the server**:
   ```bash
   python main.py
   ```

2. **Access the API documentation**:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

3. **Test with curl**:
   ```bash
   curl -X POST "http://localhost:8000/analyze" \
        -H "accept: application/json" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@sample_xray.jpg"
   ```

## 📊 API Endpoints

### Health Check
```
GET /health
```
Returns server status and loaded models.

### Analyze Image
```
POST /analyze
```
Upload and analyze a medical image.

**Parameters:**
- `file`: Image file (JPG, PNG, DICOM)
- `model_name`: Model to use (default: "chexpert")

**Response:**
```json
{
  "case_id": "uuid",
  "predictions": [
    {
      "condition": "Pneumonia",
      "confidence": 0.83,
      "rank": 1
    }
  ],
  "annotated_image_url": "/files/annotated/case_id_annotated.png",
  "model_info": {...},
  "processing_time": 2.5,
  "created_at": "2025-01-01T00:00:00Z"
}
```

### Get Case
```
GET /cases/{case_id}
```
Retrieve analysis results for a specific case.

### List Cases
```
GET /cases?limit=50&offset=0
```
List all analyzed cases with pagination.

### List Models
```
GET /models
```
Get information about available models.

## 🏗️ Architecture

```
├── src/
│   ├── api/              # FastAPI endpoints and schemas
│   ├── models/           # Model inference engine
│   ├── preprocessing/    # Image preprocessing
│   ├── visualization/    # Heatmap generation
│   └── database/         # Database models and connection
├── config/               # Configuration settings
├── storage/              # File storage
│   ├── original/         # Original uploaded images
│   └── annotated/        # Annotated images with predictions
├── tests/                # Test cases
└── main.py              # Application entry point
```

## 🔬 Supported Models

### ChestX-ray Analysis
- **Model**: DenseNet121 trained on CheXpert dataset
- **Pathologies**: 14+ conditions including:
  - Pneumonia
  - Pleural Effusion
  - Cardiomegaly
  - Pneumothorax
  - Lung Lesion
  - And more...

### Model Performance
- **Input Size**: 224x224 pixels
- **Processing Time**: < 5 seconds per image (CPU)
- **Confidence Threshold**: Configurable (default: 0.1)

## 📁 File Formats

### Supported Input Formats
- **JPEG/JPG**: Standard image format
- **PNG**: Lossless image format
- **DICOM**: Medical imaging standard (.dcm, .dicom)

### Output Formats
- **JSON**: Structured prediction results
- **PNG**: Annotated images with heatmaps

## 🛡️ Security & Compliance

⚠️ **Important**: This is a research/demo system and is **NOT medical-grade**. 

- No PHI (Protected Health Information) is stored beyond filenames
- Local-only operation for privacy
- Not HIPAA compliant (by design for MVP)
- Should not be used for actual medical diagnosis

## 🔧 Configuration

Environment variables can be set in `.env`:

```env
DEBUG=False
HOST=127.0.0.1
PORT=8000
DATABASE_URL=sqlite:///radiology.db
```

Application settings in `config/settings.py`:
- `MAX_IMAGE_SIZE`: Maximum image dimension (default: 2048px)
- `MAX_FILE_SIZE`: Maximum upload size (default: 50MB)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for predictions (default: 0.1)

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Test with sample images:
```bash
python scripts/test_api.py
```

## 📈 Performance Optimization

### CPU Performance
- Uses optimized PyTorch models
- Efficient image preprocessing
- Batch processing support

### GPU Acceleration (Optional)
If CUDA is available, models automatically use GPU acceleration for faster inference.

## 🔮 Future Enhancements

### Planned Features
- **LLM Integration**: Automated radiology report generation
- **Multi-Modal Support**: Enhanced CT/MRI analysis with MONAI
- **Web UI**: Doctor-facing interface with DICOM viewer
- **Report Generation**: PDF/DICOM output with findings
- **Model Updates**: Regular model updates and versioning

### Production Features
- **DICOM Compliance**: Full DICOM support for medical environments
- **HIPAA Compliance**: Security features for production use
- **Multi-GPU Scaling**: Distributed inference for high throughput
- **Docker Deployment**: Containerized deployment options

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚖️ Disclaimer

This software is for research and educational purposes only. It is not intended for use in medical diagnosis or treatment. Always consult with qualified healthcare professionals for medical advice.

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation in `/docs`
- Review the API documentation at `/docs` when running

---

**Built with ❤️ for advancing medical AI research**