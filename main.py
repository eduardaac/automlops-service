"""
Main Entry Point for AutoMLOps Platform

Handles:
1. Directory structure setup
2. Uvicorn server startup with the app from src.app
"""
import uvicorn
import os
from pathlib import Path

from src.app import app

IP = os.getenv('IP', '127.0.0.1')
PORT = int(os.getenv('PORT', '8000'))

def ensure_directories():
    """Ensure necessary directories exist."""
    directories = [
        "./tmp",
        "./tmp/files", 
        "./tmp/loaded_models",
        "./mlruns"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Directory created/verified: {directory}")

if __name__ == "__main__":
    print("Starting AutoMLOps API...")
    
    ensure_directories()
    
    print(f"Server starting at: http://{IP}:{PORT}")
    print(f"API docs: http://{IP}:{PORT}/docs")
    
    uvicorn.run(
        "src.app:app",
        host=IP, 
        port=PORT,
        log_level="info",
        reload=True
    )