"""
AutoMLOps API - Main Application
"""
import os
import time
import logging
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from src.database.config import init_db
from src.middleware.metrics_middleware import metrics_collector
from src.streaming.kafka_handler import kafka_handler
from src.utils.prediction_event_publisher import publicador_predicao
from src.utils.monitoring_observer import observador_monitoramento

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

NUMERO_CPUS = os.cpu_count() or 4
pool_threads = ThreadPoolExecutor(
    max_workers=min(NUMERO_CPUS * 2, 8),
    thread_name_prefix="automlops-worker"
)

publicador_predicao.register(observador_monitoramento)

@asynccontextmanager
async def lifecycle(app: FastAPI):
    """Application lifecycle manager."""
    logger.info("AutoMLOps v3.0.0 starting - Decision Support Mode...")
    
    try:
        init_db()
        logger.info("Database initialized")
        
        kafka_success = await kafka_handler.initialize()
        if kafka_success:
            logger.info("Streaming system initialized")
        
        app.state.horario_inicio = time.time()
        
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise
    
    yield
    
    logger.info("AutoMLOps shutting down...")
    
    try:
        await kafka_handler.shutdown()
        pool_threads.shutdown(wait=True)
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

app = FastAPI(
    title="AutoMLOps API",
    description="AutoML Operations Platform with Human Decision Support",
    version="3.0.0",
    lifespan=lifecycle,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.middleware("http")(metrics_collector)

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """HTTP exception handler."""
    logger.error(f"HTTP {exc.status_code}: {request.method} {request.url.path} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "erro": True,
            "mensagem": exc.detail,
            "codigo_status": exc.status_code,
            "timestamp": time.time(),
            "caminho": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """General exception handler."""
    logger.error(f"Internal error: {request.method} {request.url.path} - {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "erro": True,
            "mensagem": "Internal server error",
            "codigo_status": 500,
            "timestamp": time.time()
        }
    )

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != "your-secret-key":
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

from src.routers import training, prediction, models, monitoring
from src.routers.human_actions import router as human_actions_router

app.include_router(training.router)
app.include_router(prediction.router)
app.include_router(models.router)
app.include_router(monitoring.router)
app.include_router(human_actions_router)

@app.get("/")
async def raiz():
    """Root API endpoint."""
    return {
        "aplicacao": "AutoMLOps API",
        "versao": "3.0.0",
        "modo": "Human Decision Support",
        "descricao": "AutoML Operations Platform with alert system and manual actions",
        "status": "online",
        "timestamp": time.time(),
        "documentacao": "http://localhost:8000/docs",
        "dashboard": "http://localhost:8000/human-actions/dashboard"
    }

@app.get("/health")
async def health_check():
    """Application health check."""
    uptime = time.time() - getattr(app.state, 'horario_inicio', time.time())
    return {
        "status": "healthy",
        "mode": "human_decision_support",
        "timestamp": time.time(),
        "uptime_seconds": uptime,
        "uptime_formatted": f"{uptime/60:.1f} minutes"
    }