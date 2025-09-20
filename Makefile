# Makefile for Tele-Radiology Backend

.PHONY: help install setup run test clean dev docs

# Default target
help:
	@echo "Tele-Radiology Backend - Available Commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  make install    - Install dependencies"
	@echo "  make setup      - Complete setup (install + configure)"
	@echo ""
	@echo "Development:"
	@echo "  make run        - Start the development server"
	@echo "  make dev        - Start server in debug mode"
	@echo "  make test       - Run test suite"
	@echo "  make test-api   - Test API endpoints"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean      - Clean temporary files"
	@echo "  make docs       - Generate documentation"
	@echo "  make check      - Check system requirements"
	@echo ""

# Variables
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

# Check if virtual environment exists
check-venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "❌ Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi

# Install Python dependencies
install: check-venv
	@echo "Installing Python dependencies..."
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

# Complete setup
setup:
	@echo "Running complete setup..."
	python3 setup.py
	@echo "✅ Setup completed"

# Run the server
run: check-venv
	@echo "Starting Tele-Radiology Backend..."
	$(PYTHON) main.py

# Run in development mode
dev: check-venv
	@echo "Starting server in debug mode..."
	DEBUG=True $(PYTHON) main.py

# Run tests
test: check-venv
	@echo "Running test suite..."
	$(PYTHON) -m pytest tests/ -v

# Test API endpoints
test-api: check-venv
	@echo "Testing API endpoints..."
	$(PYTHON) scripts/test_api.py

# Example usage
example:
	@echo "Running simple example..."
	@if [ -f "sample_chest_xray.jpg" ]; then \
		python scripts/simple_example.py sample_chest_xray.jpg; \
	else \
		echo "❌ No sample image found. Create one first or provide an image path."; \
	fi

# Check system requirements
check:
	@echo "Checking system requirements..."
	@python -c "import sys; print(f'Python: {sys.version}')"
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "❌ PyTorch not installed"
	@python -c "import torchxrayvision; print('✅ TorchXRayVision available')" 2>/dev/null || echo "❌ TorchXRayVision not installed"
	@python -c "import fastapi; print('✅ FastAPI available')" 2>/dev/null || echo "❌ FastAPI not installed"

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f *.log
	@echo "✅ Cleanup completed"

# Generate documentation
docs:
	@echo "Documentation available at:"
	@echo "  📚 README.md - Main documentation"
	@echo "  🔗 docs/API.md - API reference"
	@echo "  📊 http://localhost:8000/docs - Interactive API docs (when server is running)"

# Create sample image for testing
sample-image:
	@echo "Creating sample test image..."
	python -c "
import numpy as np
from PIL import Image
# Create synthetic chest X-ray-like image
img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
# Add some structure
center = 256
for y in range(512):
    for x in range(512):
        dist = ((x-center)**2 + (y-center)**2)**0.5
        if dist < 200:
            img[y,x] = img[y,x] * 0.7
Image.fromarray(img).save('sample_chest_xray.jpg', 'JPEG')
print('✅ Created sample_chest_xray.jpg')
"

# Install in development mode
dev-install:
	@echo "Installing in development mode..."
	pip install -e .
	pip install pytest httpx
	@echo "✅ Development installation completed"

# Quick start (setup + run)
quickstart: setup run

# Full development setup
dev-setup: dev-install setup
	@echo "✅ Development environment ready!"
	@echo "Run 'make dev' to start the server in debug mode"