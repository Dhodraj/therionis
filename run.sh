#!/bin/bash
# Tele-Radiology Backend Runner Script

echo "🏥 Tele-Radiology Backend"
echo "========================="

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
    
    echo "🚀 Starting the server..."
    python main.py
else
    echo "❌ Virtual environment not found."
    echo "Please run: python3 setup.py"
    exit 1
fi