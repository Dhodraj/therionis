# API Documentation

## Tele-Radiology Backend API Reference

This document provides detailed information about all API endpoints, request/response formats, and usage examples.

## Base URL

```
http://localhost:8000
```

---

## Authentication

Currently, the API does not require authentication. This is suitable for local development and research use.

⚠️ **Security Note**: For production use, implement proper authentication and authorization.

---

## Endpoints

### 1. Health Check

Check the status and availability of the API service.

**Endpoint**: `GET /health`

**Response**: `200 OK`
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-01-01T12:00:00.000Z",
  "models_loaded": ["chexpert", "nih"],
  "device": "cpu"
}
```

**Example**:
```bash
curl -X GET "http://localhost:8000/health"
```

---

### 2. List Available Models

Get information about all loaded AI models.

**Endpoint**: `GET /models`

**Response**: `200 OK`
```json
{
  "models": {
    "chexpert": {
      "name": "DenseNet121 CheXpert",
      "pathologies": ["Pneumonia", "Effusion", "..."],
      "input_size": [224, 224],
      "description": "DenseNet121 trained on CheXpert dataset"
    },
    "nih": {
      "name": "DenseNet121 NIH",
      "pathologies": ["Pneumonia", "Effusion", "..."],
      "input_size": [224, 224],
      "description": "DenseNet121 trained on NIH ChestX-ray14 dataset"
    }
  }
}
```

---

### 3. Analyze Medical Image

Upload and analyze a medical image for pathology detection.

**Endpoint**: `POST /analyze`

**Content-Type**: `multipart/form-data`

**Parameters**:
- `file` (required): Image file (JPEG, PNG, DICOM)
- `model_name` (optional): Model to use (default: "chexpert")

**Supported File Formats**:
- JPEG/JPG
- PNG
- DICOM (.dcm, .dicom)

**File Size Limit**: 50MB

**Response**: `200 OK`
```json
{
  "case_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "predictions": [
    {
      "condition": "Pneumonia",
      "confidence": 0.83,
      "rank": 1
    },
    {
      "condition": "Pleural Effusion",
      "confidence": 0.42,
      "rank": 2
    }
  ],
  "annotated_image_url": "/files/annotated/f47ac10b-58cc-4372-a567-0e02b2c3d479_annotated.png",
  "model_info": {
    "name": "DenseNet121 CheXpert",
    "version": "chexpert",
    "total_pathologies": 14,
    "confidence_threshold": 0.1
  },
  "processing_time": 2.45,
  "created_at": "2025-01-01T12:00:00.000Z"
}
```

**Example**:
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@chest_xray.jpg" \
     -F "model_name=chexpert"
```

**Error Responses**:

`400 Bad Request` - Invalid file format or size:
```json
{
  "error": "Unsupported format. Supported: .jpg, .jpeg, .png, .dcm, .dicom"
}
```

`500 Internal Server Error` - Analysis failed:
```json
{
  "error": "Analysis failed: Model inference error"
}
```

---

### 4. Retrieve Case Information

Get detailed information about a previously analyzed case.

**Endpoint**: `GET /cases/{case_id}`

**Path Parameters**:
- `case_id`: Unique case identifier (UUID)

**Response**: `200 OK`
```json
{
  "case_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "original_filename": "chest_xray.jpg",
  "predictions": [
    {
      "condition": "Pneumonia",
      "confidence": 0.83,
      "rank": 1
    }
  ],
  "annotated_image_url": "/files/annotated/f47ac10b-58cc-4372-a567-0e02b2c3d479_annotated.png",
  "original_image_url": "/files/original/f47ac10b-58cc-4372-a567-0e02b2c3d479_chest_xray.jpg",
  "model_info": {
    "name": "DenseNet121 CheXpert",
    "version": "chexpert",
    "total_pathologies": 14,
    "confidence_threshold": 0.1
  },
  "processing_time": 2.45,
  "created_at": "2025-01-01T12:00:00.000Z",
  "file_size": 245760,
  "image_format": "jpg"
}
```

**Example**:
```bash
curl -X GET "http://localhost:8000/cases/f47ac10b-58cc-4372-a567-0e02b2c3d479"
```

**Error Response**:
`404 Not Found`:
```json
{
  "error": "Case f47ac10b-58cc-4372-a567-0e02b2c3d479 not found"
}
```

---

### 5. List All Cases

Retrieve a paginated list of all analyzed cases.

**Endpoint**: `GET /cases`

**Query Parameters**:
- `limit` (optional): Maximum number of cases to return (default: 50, max: 100)
- `offset` (optional): Number of cases to skip (default: 0)

**Response**: `200 OK`
```json
[
  {
    "case_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "original_filename": "chest_xray.jpg",
    "predictions": [...],
    "annotated_image_url": "/files/annotated/...",
    "original_image_url": "/files/original/...",
    "model_info": {...},
    "processing_time": 2.45,
    "created_at": "2025-01-01T12:00:00.000Z",
    "file_size": 245760,
    "image_format": "jpg"
  }
]
```

**Example**:
```bash
curl -X GET "http://localhost:8000/cases?limit=10&offset=0"
```

---

### 6. Access Image Files

Download original or annotated images.

**Original Images**: `GET /files/original/{case_id}_{filename}`

**Annotated Images**: `GET /files/annotated/{case_id}_annotated.png`

**Example**:
```bash
# Download original image
curl -X GET "http://localhost:8000/files/original/f47ac10b-58cc-4372-a567-0e02b2c3d479_chest_xray.jpg" \
     --output original_image.jpg

# Download annotated image
curl -X GET "http://localhost:8000/files/annotated/f47ac10b-58cc-4372-a567-0e02b2c3d479_annotated.png" \
     --output annotated_image.png
```

---

## Data Models

### PredictionResponse
```json
{
  "condition": "string",      // Name of detected condition
  "confidence": 0.85,         // Confidence score (0-1)
  "rank": 1                   // Ranking by confidence
}
```

### ModelInfo
```json
{
  "name": "string",                    // Human-readable model name
  "version": "string",                 // Model version identifier
  "total_pathologies": 14,             // Number of detectable conditions
  "confidence_threshold": 0.1          // Minimum confidence threshold
}
```

---

## Error Handling

All errors follow a consistent format:

```json
{
  "error": "Brief error description",
  "detail": "Detailed error information (optional)",
  "timestamp": "2025-01-01T12:00:00.000Z"
}
```

### Common HTTP Status Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

---

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider implementing rate limiting based on your requirements.

---

## CORS

Cross-Origin Resource Sharing (CORS) is enabled for all origins during development. Adjust this for production use.

---

## Interactive Documentation

The API provides interactive documentation through:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to:
- Explore all endpoints
- Test API calls directly
- View detailed schemas
- Download OpenAPI specifications

---

## SDK and Client Libraries

### Python Example
```python
import requests

# Analyze an image
with open('chest_xray.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/analyze', files=files)
    result = response.json()
    print(f"Case ID: {result['case_id']}")
```

### JavaScript Example
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/analyze', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log('Case ID:', data.case_id));
```

### cURL Examples

See individual endpoint documentation above for cURL examples.

---

## Performance Considerations

### Processing Times
- Typical processing time: 2-5 seconds per image
- Factors affecting speed:
  - Image size and resolution
  - Model complexity
  - Hardware (CPU vs GPU)

### Optimization Tips
- Resize large images before upload
- Use GPU acceleration when available
- Consider batch processing for multiple images

---

## Troubleshooting

### Common Issues

1. **"Model not found" error**
   - Check `/health` endpoint to see loaded models
   - Ensure TorchXRayVision models downloaded correctly

2. **"File too large" error**
   - Maximum file size is 50MB
   - Resize or compress images before upload

3. **"Unsupported format" error**
   - Supported formats: JPEG, PNG, DICOM
   - Check file extension and content

4. **Slow processing**
   - Install PyTorch with GPU support if available
   - Check system resources (RAM, CPU usage)

### Logs and Debugging

- Check server logs for detailed error information
- Enable debug mode in configuration for verbose logging
- Use the `/health` endpoint to verify system status