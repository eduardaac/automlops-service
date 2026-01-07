from pydantic import BaseSettings
import os

class Settings(BaseSettings):
    """Centralized application settings.
    
    Values can be overridden via environment variables or .env file.
    """
    database_url: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://automlops_user:1598@127.0.0.1:5432/automlops_db"
    )
    
    kafka_bootstrap_servers: str = os.getenv("KAFKA_SERVERS", "localhost:9092")
    
    api_key: str = os.getenv("API_KEY", "your-secret-key")
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024))
    
    ip: str = os.getenv("IP", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    
    # MLflow URI - padrão local, mas pode ser sobrescrito via env
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()