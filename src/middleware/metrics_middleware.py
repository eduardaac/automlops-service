"""
AutoMLOps metrics collection middleware with Prometheus
"""
import time
import logging
from fastapi import Request, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, Info
from typing import Callable

logger = logging.getLogger(__name__)

http_requests_total = Counter(
    'automlops_http_requests_total',
    'Total AutoMLOps HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'automlops_http_request_duration_seconds',
    'AutoMLOps HTTP request duration',
    ['method', 'endpoint']
)

predictions_total = Counter(
    'automlops_predictions_total',
    'Total predictions made',
    ['model_id', 'model_name', 'model_type']
)

prediction_duration_seconds = Histogram(
    'automlops_prediction_duration_seconds',
    'Prediction duration in seconds',
    ['model_id', 'model_name'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

active_requests = Gauge(
    'automlops_active_requests',
    'Number of active requests'
)

model_accuracy = Gauge(
    'automlops_model_accuracy',
    'Current model accuracy',
    ['model_id', 'model_name', 'model_type', 'stage']
)

model_precision = Gauge(
    'automlops_model_precision',
    'Current model precision',
    ['model_id', 'model_name', 'model_type', 'stage']
)

model_recall = Gauge(
    'automlops_model_recall',
    'Current model recall',
    ['model_id', 'model_name', 'model_type', 'stage']
)

model_f1_score = Gauge(
    'automlops_model_f1_score',
    'Current model F1-Score',
    ['model_id', 'model_name', 'model_type', 'stage']
)

data_drift_score = Gauge(
    'automlops_data_drift_score',
    'Detected data drift score',
    ['dataset_name', 'feature']
)

training_jobs_total = Counter(
    'automlops_training_jobs_total',
    'Total training jobs',
    ['status']
)

active_models = Gauge(
    'automlops_active_models',
    'Number of active models',
    ['stage']
)

def track_training_job(status: str):
    """Register training job"""
    training_jobs_total.labels(status=status).inc()
    logger.info(f"Training job tracked: {status}")

def update_model_accuracy(model_id: str, model_name: str, model_type: str, stage: str, accuracy: float):
    """Update accuracy of a specific model"""
    model_accuracy.labels(
        model_id=model_id, 
        model_name=model_name, 
        model_type=model_type, 
        stage=stage
    ).set(accuracy)
    logger.info(f"Model accuracy updated: {model_name} - {accuracy}")

def update_model_metrics(model_id: str, model_name: str, model_type: str, stage: str, metrics: dict):
    """Update all metrics of a model"""
    
    model_accuracy.labels(
        model_id=model_id, 
        model_name=model_name, 
        model_type=model_type, 
        stage=stage
    ).set(metrics.get('accuracy', 0))
    
    model_precision.labels(
        model_id=model_id, 
        model_name=model_name, 
        model_type=model_type, 
        stage=stage
    ).set(metrics.get('precision', 0))
    
    model_recall.labels(
        model_id=model_id, 
        model_name=model_name, 
        model_type=model_type, 
        stage=stage
    ).set(metrics.get('recall', 0))
    
    model_f1_score.labels(
        model_id=model_id, 
        model_name=model_name, 
        model_type=model_type, 
        stage=stage
    ).set(metrics.get('f1_score', 0))
    
    active_models.labels(stage=stage).inc()
    
    logger.info(f"Model metrics updated: {model_name} - {stage}")

def track_prediction(model_id: str, model_name: str, model_type: str, duration: float):
    """Register a prediction"""
    predictions_total.labels(
        model_id=model_id,
        model_name=model_name,
        model_type=model_type
    ).inc()
    
    prediction_duration_seconds.labels(
        model_id=model_id,
        model_name=model_name
    ).observe(duration)

def track_data_drift(dataset_name: str, feature: str, drift_score: float):
    """Register data drift"""
    data_drift_score.labels(
        dataset_name=dataset_name,
        feature=feature
    ).set(drift_score)

def track_model_promotion(from_stage: str, to_stage: str, model_name: str):
    """Register model promotion"""
    logger.info(f"Model promotion: {model_name} from {from_stage} to {to_stage}")

async def metrics_collector(request: Request, call_next: Callable) -> Response:
    """Main middleware for automatic metrics collection"""
    start_time = time.time()
    
    active_requests.inc()
    
    try:
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        http_requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).inc()
        
        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        return response
        
    finally:
        active_requests.dec()

metrics_middleware = metrics_collector

def get_metrics():
    """Return metrics in Prometheus format"""
    return generate_latest()
