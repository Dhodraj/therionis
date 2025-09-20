#!/usr/bin/env python3
"""
Main entry point for the Tele-Radiology Backend
"""
import sys
import os
import uvicorn
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.api.main import app


def main():
    """
    Main function to start the Tele-Radiology Backend server
    """
    print("=" * 60)
    print("🏥 Tele-Radiology Backend Starting...")
    print("=" * 60)
    
    # Print configuration
    print(f"📁 Storage directory: {settings.storage_dir}")
    print(f"🗄️  Database: {settings.database_url}")
    print(f"🌐 Host: {settings.host}:{settings.port}")
    print(f"🔧 Debug mode: {settings.debug}")
    print("=" * 60)
    
    # Start the server
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
        access_log=True
    )


if __name__ == "__main__":
    main()