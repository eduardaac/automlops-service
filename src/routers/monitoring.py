"""
Monitoring endpoints router
"""
import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, File, UploadFile, Response
from sqlalchemy.orm import Session
from pathlib import Path

from src.database.config import get_db
from src.schemas.monitoring import RespostaDrift, StatusSistema, MetricasMonitoramento
from src.services.file_service import FileService
from src.services.alert_service import alert_service
from src.services.performance_service import performance_service
from src.utils.check_data_drift import check_data_drift
from src.middleware.metrics_middleware import get_metrics
from src.streaming.kafka_handler import kafka_handler
from prometheus_client import CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

@router.get("/metrics/prometheus", summary="Expose Prometheus-compatible metrics")
async def prometheus_metrics():
    """
    ### Description
    Exposes system and ML model metrics in Prometheus text-based exposition format.
    
    ### Metrics Included
    API request rates, latencies, prediction counts and durations, training statistics, 
    error rates, HTTP status codes, and resource utilization.
    
    ### Response Format
    Prometheus text format with Counter (cumulative), Gauge (current), and Histogram (distribution) metric types.
    """
    try:
        metrics_content = get_metrics()
        return Response(
            content=metrics_content,
            media_type=CONTENT_TYPE_LATEST
        )
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return Response(
            content="# Error generating metrics\n",
            media_type=CONTENT_TYPE_LATEST,
            status_code=500
        )

@router.get("/health", response_model=StatusSistema, summary="Check overall system health status")
async def verify_system_health():
    """
    ### Description
    Performs health check of all system components including message broker, database, and API services.
    
    ### Response
    Returns overall health status (healthy/unhealthy), execution timestamp, and individual component 
    status map. Status is healthy when all components are operational and responsive.
    """
    try:
        kafka_status = await kafka_handler.health_check()
        
        return StatusSistema(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            componentes={
                "kafka": "healthy" if kafka_status else "unhealthy",
                "database": "healthy",
                "api": "healthy"
            }
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check error: {str(e)}"
        )
