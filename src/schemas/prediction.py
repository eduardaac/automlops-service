"""
Schemas for prediction endpoints
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime

class PredictionRequest(BaseModel):
    """Schema for individual prediction request"""
    data: Dict[str, Any] = Field(..., description="Input data", min_length=1)
    model_key: Optional[str] = Field(None, description="Model key", max_length=50)
    
    @validator('data')
    def validate_data_not_empty(cls, v):
        if not v:
            raise ValueError('Data cannot be empty')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": {
                    "feature_1": 125.5,
                    "feature_2": 45.2,
                    "feature_3": 3.8
                },
                "model_key": "abc123def456"
            }
        }

class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request"""
    data_list: List[Dict[str, Any]] = Field(..., description="List of data for prediction")
    model_key: Optional[str] = Field(None, description="Specific model key")
    
    class Config:
        json_schema_extra = {
            "example": {
                "data_list": [
                    {"feature_1": 125.5, "feature_2": 45.2}
                ],
                "model_key": "abc123"
            }
        }

class BatchComparisonRequest(BaseModel):
    """Schema for batch comparison request with optional labels"""
    features: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries for prediction")
    labels: Optional[List[Any]] = Field(None, description="Optional true labels for evaluation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [
                    {
                        "feature_1": 125.5,
                        "feature_2": 45.2,
                        "feature_3": 3.8,
                        "feature_4": 1200,
                        "feature_5": 0.85
                    },
                    {
                        "feature_1": 130.2,
                        "feature_2": 48.7,
                        "feature_3": 4.1,
                        "feature_4": 1150,
                        "feature_5": 0.92
                    }
                ],
                "labels": [1, 0]
            }
        }

class PredictionResponse(BaseModel):
    """Schema for individual prediction response (classification and regression)"""
    prediction: Any = Field(
        ..., 
        description="Predicted value - Class label (classification) or numeric value (regression)"
    )
    model_key: str = Field(..., description="Model key used for prediction")
    model_name: str = Field(..., description="Algorithm name (e.g., RandomForestClassifier, XGBRegressor)")
    confidence: Optional[float] = Field(
        None, 
        description="Prediction confidence (0-1): Probability for classification, pseudo-confidence for regression"
    )
    probabilities: Optional[List[float]] = Field(
        None, 
        description="Class probabilities (classification only) - None for regression models"
    )
    health_index: Optional[float] = Field(
        None, 
        description="🏭 Model Health Index (0.0=critical, 1.0=perfect): Combines drift (40%) + confidence (50%) + anomaly (10%)"
    )
    timestamp: str = Field(..., description="Prediction timestamp (ISO 8601 format)")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "model_key": "abc123def456",
                "model_name": "RandomForestClassifier",
                "confidence": 0.85,
                "probabilities": [0.15, 0.85],
                "health_index": 0.8542,
                "timestamp": "2024-12-07T10:30:00"
            }
        }

class BatchPredictionResult(BaseModel):
    """Schema for individual result within batch (classification and regression)"""
    prediction: Any = Field(
        ..., 
        description="Predicted value - Class label (classification) or numeric value (regression)"
    )
    confidence: Optional[float] = Field(
        None, 
        description="Confidence score (0-1): Probability for classification, pseudo-confidence for regression"
    )
    probabilities: Optional[List[float]] = Field(
        None, 
        description="Class probabilities (classification only) - None for regression"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 1,
                "confidence": 0.85,
                "probabilities": [0.15, 0.85]
            }
        }

class ModelPerformanceMetrics(BaseModel):
    """
    Model performance metrics for production monitoring.
    
    🏭 HEALTH INDEX is the PRIMARY metric for operational decisions.
    Combines: Data Drift (40%) + Confidence (50%) + Anomaly Detection (10%)
    
    Supports both classification and regression models.
    """
    confidence_avg: float = Field(
        ..., 
        description="Average prediction confidence across batch (0.0-1.0)"
    )
    low_confidence_count: int = Field(
        ..., 
        description="Count of predictions with confidence < 0.7 (quality indicator)"
    )
    health_index: float = Field(
        ..., 
        description="🏭 Health Index (0.0-1.0): PRIMARY operational metric - >0.7=healthy, 0.5-0.7=warning, <0.5=critical"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "confidence_avg": 0.9233,
                "low_confidence_count": 0,
                "health_index": 0.8542
            }
        }

class ModelComparisonResult(BaseModel):
    """Schema for model comparison result"""
    model_key: str = Field(..., description="Model key")
    model_name: str = Field(..., description="Model name")
    stage: str = Field(..., description="Model stage (champion/challenger/archived)")
    predictions: List[BatchPredictionResult] = Field(..., description="Prediction results")
    performance_metrics: ModelPerformanceMetrics = Field(..., description="Performance metrics")
    total_predictions: int = Field(..., description="Total predictions made")

class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response with comparison"""
    results: List[ModelComparisonResult] = Field(..., description="Results per model")
    total_models: int = Field(..., description="Total models used")
    timestamp: str = Field(..., description="Operation timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "model_key": "abc123def456",
                        "model_name": "RandomForestClassifier",
                        "stage": "champion",
                        "predictions": [
                            {
                                "prediction": 1,
                                "confidence": 0.85,
                                "probabilities": [0.15, 0.85]
                            }
                        ],
                        "performance_metrics": {
                            "accuracy": 0.87,
                            "precision": 0.85,
                            "recall": 0.89,
                            "f1_score": 0.87,
                            "confidence_avg": 0.82,
                            "low_confidence_count": 2,
                            "health_index": 0.8542
                        },
                        "total_predictions": 10
                    }
                ],
                "total_models": 3,
                "timestamp": "2024-01-15T10:30:00"
            }
        }

class ActiveModelsResponse(BaseModel):
    """Schema para resposta de modelos ativos"""
    active_models: List[Dict[str, Any]] = Field(..., description="Lista de modelos ativos")
    total: int = Field(..., description="Total de modelos ativos")
    
    class Config:
        schema_extra = {
            "example": {
                "active_models": [
                    {
                        "key": "abc123def456",
                        "name": "RandomForestClassifier",
                        "stage": "champion",
                        "accuracy": 0.87,
                        "created_at": "2024-01-15T09:00:00"
                    }
                ],
                "total": 3
            }
        }