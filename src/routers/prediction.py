"""
Prediction endpoints router
"""
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Path, Body
from sqlalchemy.orm import Session

from src.database.config import get_db
from src.database.models.Result import Result as DBResult
from src.database.models.PerformanceLog import PerformanceLog
from src.services.prediction_service import prediction_service
from src.schemas.prediction import ActiveModelsResponse, BatchComparisonRequest
from src.utils.prediction_event_publisher import publicador_predicao
from src.middleware.metrics_middleware import track_prediction

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])

def sanitize_float(value: float) -> float:
    """
    Sanitize float values to ensure JSON compatibility.
    Converts inf, -inf, and NaN to safe values.
    
    Args:
        value: Float value to sanitize
        
    Returns:
        Safe float value (0.0 for NaN/inf)
    """
    if not isinstance(value, (int, float)):
        return value
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return value

def sanitize_metrics(metrics):
    """
    Recursively sanitize all numeric values in a data structure.
    Handles dicts, lists, and numeric values.
    
    Args:
        metrics: Data structure to sanitize (dict, list, or value)
        
    Returns:
        Sanitized data structure
    """
    if isinstance(metrics, dict):
        return {
            key: sanitize_metrics(value)
            for key, value in metrics.items()
        }
    elif isinstance(metrics, list):
        return [sanitize_metrics(item) for item in metrics]
    elif isinstance(metrics, (int, float, np.number)):
        return sanitize_float(metrics)
    else:
        return metrics

@router.post(
    "/batch/{model_key}",
    summary="Generate batch predictions with health monitoring"
)
async def batch_prediction(
    model_key: str = Path(
        ...,
        description="Unique identifier of the model to use for predictions",
        example="a66ccf8aa3634dfb5afe46cb19f79d01"
    ),
    data_list: List[Dict[str, Any]] = Body(
        ...,
        description="Array of input feature dictionaries for batch prediction. Each dictionary must contain all required feature columns matching the trained model schema.",
        example=[
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
        ]
    ),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    ### Description
    Generates batch predictions with comprehensive health monitoring for production model oversight.
    Supports both **classification** and **regression** models automatically.
    
    ### Request Body
    Array of feature dictionaries containing all required columns matching the model's training schema.
    Feature names and types must match exactly what the model was trained on.
    
    ### Response
    Returns:
    - **predictions**: Array of predicted values (classes or numeric values)
    - **confidence**: Confidence scores for each prediction
      - **Classification**: Based on probability of predicted class
      - **Regression**: Pseudo-confidence based on prediction variance
    - **health_index**: Model health score (0.0-1.0) combining:
      - Data drift (40%): Distribution changes in input features
      - Confidence (50%): Prediction confidence levels
      - Anomaly detection (10%): Statistical outlier detection
    - **action_recommended**: Suggested action based on health status
    
    ### Health Status Interpretation
    - **> 0.70**: ✅ Healthy - Model operating normally
    - **0.50-0.70**: ⚠️ Warning - Consider switching to challenger model
    - **< 0.50**: 🚨 Critical - Urgent retraining recommended
    
    ### Model Type Support
    - **✅ Classification**: Binary, multiclass (probabilities + confidence)
    - **✅ Regression**: Continuous values (pseudo-confidence)
    """
    try:
        from src.utils.monitoring_observer import observador_monitoramento
        if hasattr(observador_monitoramento, 'set_background_tasks'):
            observador_monitoramento.set_background_tasks(background_tasks)
        
        start_time = time.time()
        result = await prediction_service.predict_batch(model_key, data_list)
        duration = time.time() - start_time
        
        # Busca informações do modelo
        model = db.query(DBResult).filter(DBResult.key == model_key).first()
        if model:
            track_prediction(
                model_id=model_key[:8],
                model_name=model.model,
                model_type=model.model,
                duration=duration
            )
        
        import pandas as pd
        
        # Extrair predictions, probabilities e metadados do resultado
        predictions = [p.get("prediction") for p in result.get("predictions", [])]
        probabilities = [p.get("probabilities", []) for p in result.get("predictions", [])]
        confidences = [p.get("confidence") for p in result.get("predictions", [])]
        
        # Detectar tipo de modelo
        is_classification = probabilities and all(p is not None for p in probabilities if p)
        pseudo_confidences = None if is_classification else confidences
        
        health_analysis = await _unified_prediction_health_analysis(
            db=db,
            model_key=model_key,
            predictions=predictions,
            probabilities=probabilities if is_classification else [],
            pseudo_confidences=pseudo_confidences,
            is_classification=is_classification,
            input_data=pd.DataFrame(data_list),
            evaluation_type="batch_prediction",
            trigger_alerts=True  # Trigger alerts only once at the end
        )
        
        # Adicionar health_index e action_recommended à resposta
        result["health_index"] = health_analysis.get("health_index")
        result["action_recommended"] = health_analysis.get("action_recommended")
        
        if health_analysis.get("needs_action"):
            logger.warning(
                f"Action needed for model {model_key[:8]}: "
                f"{health_analysis.get('action_recommended')}"
            )
        
        logger.info(
            f"Batch predictions made - Model: {model_key[:8]}, Total: {len(data_list)} | "
            f"Health: {health_analysis.get('health_index', 'N/A')}"
        )
        return result
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post(
    "/batch-comparison",
    summary="Compare predictions across all active models with optional performance evaluation"
)
async def batch_comparison_prediction(
    request: BatchComparisonRequest,
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    # Entry point - validate received data
    logger.info(f"[ENDPOINT] batch_comparison_prediction called")
    logger.info(f"[ENDPOINT] Request received - features: {len(request.features)}, labels: {request.labels}")
    
    # Convert Pydantic model to dict
    data = {
        "features": request.features,
        "labels": request.labels
    }
    logger.info(f"[ENDPOINT] Data converted for processing")
    logger.info(f"🔍 [ENDPOINT] Keys em 'data': {list(data.keys())}")
    """
    ### Description
    Compares predictions across all active models (Champion + Challengers) to support model governance
    and A/B testing. Provides side-by-side performance evaluation of multiple models.
    
    ### Request Body
    - **features**: Array of feature dictionaries (required) - Must match training schema
    - **labels**: Array of true values (optional) - For performance validation:
      - **Classification**: True class labels (e.g., [0, 1, 1, 0])
      - **Regression**: True numeric values (e.g., [125.3, 130.8, 142.1])
    
    ### Response
    Returns array of results, one per active model, containing:
    - **model_info**: Model metadata (key, name, stage, training metrics)
    - **predictions**: Predicted values from this model
    - **confidence_metrics**: Average confidence and low confidence count
    - **health_index**: Model health score (0.0-1.0)
    - **real_metrics** (when labels provided):
      - **Classification**: Accuracy, Precision, Recall, F1-Score
      - **Regression**: MAE, RMSE, R², MSE
    - **duration**: Prediction time in seconds
    
    ### Use Cases
    1. **A/B Testing**: Compare Champion vs Challengers without labels
    2. **Model Validation**: Evaluate real performance with labeled test data
    3. **Model Selection**: Identify best performer before promotion
    4. **Performance Monitoring**: Track degradation across model lineage
    
    ### Model Type Support
    - **✅ Classification**: Binary and multiclass problems
    - **✅ Regression**: Continuous value prediction
    """
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
        from src.utils.monitoring_observer import observador_monitoramento
        import pandas as pd
        
        if hasattr(observador_monitoramento, 'set_background_tasks'):
            observador_monitoramento.set_background_tasks(background_tasks)
        
        # 🔍 LOG 2: Extração de features e labels
        logger.info(f"🔍 [PROCESSAMENTO] Extraindo 'features' de data...")
        data_list = data.get("features", [])
        logger.info(f"🔍 [PROCESSAMENTO] Features extraídas: {len(data_list)} registros")
        logger.info(f"🔍 [PROCESSAMENTO] Tipo de data_list: {type(data_list)}")
        
        logger.info(f"🔍 [PROCESSAMENTO] Extraindo 'labels' de data...")
        y_true = data.get("labels", None)
        logger.info(f"🔍 [PROCESSAMENTO] Labels: {y_true}")
        
        # 🔍 LOG 3: Validação de dados
        logger.info(f"🔍 [VALIDAÇÃO] Validando dados de entrada...")
        if not data_list:
            logger.error(f"❌ [VALIDAÇÃO] Features array está vazio!")
            raise HTTPException(status_code=400, detail="Features array is required")
        
        logger.info(f"✅ [VALIDAÇÃO] Features OK: {len(data_list)} registros")
        
        if y_true is not None and len(data_list) != len(y_true):
            logger.error(f"❌ [VALIDAÇÃO] Tamanhos incompatíveis - Features: {len(data_list)}, Labels: {len(y_true)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Features ({len(data_list)}) and labels ({len(y_true)}) must have same length"
            )
        
        logger.info(f"✅ [VALIDAÇÃO] Validação completa - prosseguindo...")
        
        has_labels = y_true is not None
        evaluation_type = "labeled_evaluation" if has_labels else "batch_comparison"
        
        active_models = db.query(DBResult).filter(
            DBResult.is_active == True
        ).all()
        
        if not active_models:
            raise HTTPException(
                status_code=404,
                detail="No active models found"
            )
        
        comparison_results = []
        
        for model in active_models:
            try:
                start_time = time.time()
                result = await prediction_service.predict_batch(
                    model.key, 
                    data_list
                )
                duration = time.time() - start_time
                
                # Rastreia predição no Prometheus
                track_prediction(
                    model_id=model.key[:8],
                    model_name=model.model,
                    model_type=model.model,
                    duration=duration
                )
                
                # Extrair predictions e probabilities do resultado
                predictions = [p.get("prediction") for p in result.get("predictions", [])]
                probabilities = [p.get("probabilities", []) for p in result.get("predictions", [])]
                
                # Calculate real metrics if labels are provided
                real_metrics = None
                if has_labels:
                    # Check if model is classification or regression
                    is_classification = hasattr(model, 'predict_proba') or all(isinstance(p, (int, np.integer)) for p in predictions[:5])
                    
                    if is_classification:
                        # Classification metrics
                        accuracy = accuracy_score(y_true, predictions)
                        precision = precision_score(y_true, predictions, average='weighted', zero_division=0)
                        recall = recall_score(y_true, predictions, average='weighted', zero_division=0)
                        f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
                        
                        real_metrics = {
                            "accuracy": round(accuracy, 4),
                            "precision": round(precision, 4),
                            "recall": round(recall, 4),
                            "f1_score": round(f1, 4)
                        }
                        
                        logger.info(
                            f"✅ Model evaluated (Classification): {model.key[:8]} ({model.stage.value}) - "
                            f"Accuracy: {accuracy:.4f} | F1: {f1:.4f}"
                        )
                    else:
                        # Regression metrics
                        mae = mean_absolute_error(y_true, predictions)
                        mse = mean_squared_error(y_true, predictions)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_true, predictions)
                        
                        # Sanitize values to prevent JSON serialization errors
                        real_metrics = {
                            "mae": sanitize_float(round(mae, 4)),
                            "mse": sanitize_float(round(mse, 4)),
                            "rmse": sanitize_float(round(rmse, 4)),
                            "r2": sanitize_float(round(r2, 4))
                        }
                        
                        logger.info(
                            f"✅ Model evaluated (Regression): {model.key[:8]} ({model.stage.value}) - "
                            f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}"
                        )
                
                # Detect model type for health analysis
                # If we have labels and calculated real_metrics, use that info
                # Otherwise, check if probabilities are present (classification models return probabilities)
                if has_labels and real_metrics:
                    is_classification_model = "accuracy" in real_metrics
                else:
                    # Detect by checking if model returned probabilities (classification) or not (regression)
                    is_classification_model = probabilities and len(probabilities) > 0 and probabilities[0] is not None
                
                logger.info(
                    f"Model {model.key[:8]} detected as: "
                    f"{'Classification' if is_classification_model else 'Regression'} "
                    f"(has_labels={has_labels}, has_probs={bool(probabilities)})"
                )
                
                # 🏥 UNIFIED HEALTH ANALYSIS
                health_analysis = await _unified_prediction_health_analysis(
                    db=db,
                    model_key=model.key,
                    predictions=predictions,
                    probabilities=probabilities,
                    input_data=pd.DataFrame(data_list),
                    evaluation_type=evaluation_type,
                    trigger_alerts=(not has_labels),  # Don't trigger alerts during evaluation
                    is_classification=is_classification_model,
                    real_metrics=real_metrics  # Pass real metrics to be saved in PerformanceLog
                )
                
                health_index = health_analysis.get("health_index", 0.5)
                performance_metrics = health_analysis.get("performance_metrics", {})
                
                # Ensure health_index is in performance_metrics
                if health_index is not None:
                    performance_metrics["health_index"] = round(health_index, 4)
                
                logger.info(
                    f"✅ Performance metrics for {model.key[:8]}: "
                    f"confidence_avg={performance_metrics.get('confidence_avg', 0):.4f}, "
                    f"health_index={performance_metrics.get('health_index', 0):.4f}"
                )
                
                if health_analysis.get("needs_action") and not has_labels:
                    logger.warning(
                        f"⚠️ Action needed for model {model.key[:8]} ({model.stage.value}): "
                        f"{health_analysis.get('action_recommended')}"
                    )
                
                model_result = {
                    "model_key": model.key,
                    "model_name": model.model,
                    "stage": model.stage.value,
                    "predictions": result["predictions"],
                    "performance_metrics": performance_metrics,
                    "total_predictions": len(data_list)
                }
                
                # Add real metrics if available
                if real_metrics:
                    model_result["real_metrics"] = real_metrics
                
                comparison_results.append(model_result)
                
            except Exception as e:
                logger.error(f"Error in prediction with model {model.key[:8]}: {e}")
                continue
        
        if not comparison_results:
            raise HTTPException(
                status_code=500,
                detail="All model predictions failed"
            )
        
        # Commit performance logs if labels were provided
        if has_labels:
            db.commit()
            logger.info(f"Performance metrics recorded for {len(comparison_results)} models")
        
        logger.info(
            f"Batch comparison done with {len(comparison_results)} models "
            f"({'with evaluation' if has_labels else 'without evaluation'})"
        )
        
        # Retorna dicionário simples - sem usar Pydantic schema
        response_dict = {
            "results": comparison_results,
            "total_models": len(comparison_results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Sanitize all numeric values to prevent JSON serialization errors
        response_dict = sanitize_metrics(response_dict)
        
        # Add evaluation flag to response if needed
        if has_labels:
            response_dict["evaluation_performed"] = True
            response_dict["sample_size"] = len(data_list)
        
        return response_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _estimate_traditional_metrics(
    health_index: float,
    confidence_avg: float,
    low_confidence_count: int,
    total_predictions: int,
    is_classification: bool = True
) -> Dict[str, float]:
    """
    Estimates traditional metrics based on Health Index and confidence.
    
    Uses the same philosophy as Health Index: combines multiple indicators
    to estimate model performance without requiring labels.
    
    CLASSIFICATION Estimation formulas:
    - Accuracy = Health Index (strong correlation with overall health)
    - Precision = Average confidence (high confidence indicates precision)
    - Recall = Health Index adjusted by low confidence volume
    - F1 Score = Harmonic mean of estimated Precision and Recall
    - ROC AUC = Health Index with confidence boost
    
    REGRESSION Estimation:
    - Cannot reliably estimate MAE/RMSE/R² without labels
    - Returns None for regression metrics (will not be saved)
    
    Args:
        health_index: Model health index (0.0-1.0)
        confidence_avg: Average prediction confidence
        low_confidence_count: Number of low confidence predictions
        total_predictions: Total predictions
        is_classification: True for classification, False for regression
    
    Returns:
        Dictionary with estimated metrics (classification) or empty dict (regression)
    """
    # For REGRESSION: Cannot estimate MAE/RMSE/R² without real labels
    # Return empty dict - metrics will only be saved when labels are provided
    if not is_classification:
        return {}
    
    # For CLASSIFICATION: Estimate metrics based on confidence and health
    low_conf_ratio = low_confidence_count / total_predictions if total_predictions > 0 else 0.0
    
    # 1. Accuracy: directly correlated with Health Index
    # High health = high accuracy (with slight confidence boost)
    estimated_accuracy = (health_index * 0.85) + (confidence_avg * 0.15)
    estimated_accuracy = min(max(estimated_accuracy, 0.0), 1.0)  # Clamp [0, 1]
    
    # 2. Precision: strongly correlated with confidence
    # High confidence = high precision (model is confident about positive predictions)
    estimated_precision = (confidence_avg * 0.90) + (health_index * 0.10)
    estimated_precision = min(max(estimated_precision, 0.0), 1.0)
    
    # 3. Recall: Health Index penalized by low confidence
    # If there are many low confidence predictions, recall may be compromised
    recall_penalty = low_conf_ratio * 0.15  # Up to 15% penalty
    estimated_recall = health_index - recall_penalty
    estimated_recall = min(max(estimated_recall, 0.0), 1.0)
    
    # 4. F1 Score: harmonic mean of precision and recall
    if estimated_precision + estimated_recall > 0:
        estimated_f1 = 2 * (estimated_precision * estimated_recall) / (estimated_precision + estimated_recall)
    else:
        estimated_f1 = 0.0
    
    # 5. ROC AUC: Health Index with confidence boost
    # Combines overall health with separation capability (confidence)
    estimated_roc_auc = (health_index * 0.75) + (confidence_avg * 0.25)
    estimated_roc_auc = min(max(estimated_roc_auc, 0.5), 1.0)  # Minimum 0.5 (random)
    
    return {
        "accuracy": round(estimated_accuracy, 4),
        "precision": round(estimated_precision, 4),
        "recall": round(estimated_recall, 4),
        "f1_score": round(estimated_f1, 4),
        "roc_auc": round(estimated_roc_auc, 4)
    }


async def _unified_prediction_health_analysis(
    db: Session,
    model_key: str,
    predictions: List[Any],
    probabilities: List[List[float]],
    input_data: Any,
    evaluation_type: str = "prediction",
    trigger_alerts: bool = True,
    is_classification: bool = True,
    pseudo_confidences: List[float] = None,
    real_metrics: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    UNIFIED HEALTH ANALYSIS FOR ALL PREDICTIONS (Classification & Regression)
    
    Calculates health_index, records to PerformanceLog, and detects need 
    for retraining or model switching.
    
    Args:
        db: Database session
        model_key: Model key
        predictions: List of predictions
        probabilities: List of probabilities per prediction (classification only)
        pseudo_confidences: List of pseudo-confidences (regression only)
        is_classification: True for classification, False for regression
        input_data: Input data (DataFrame or list of dicts)
        evaluation_type: Evaluation type (batch_prediction, batch_comparison)
        trigger_alerts: If True, triggers alerts when health degrades
        real_metrics: Dict with real metrics (accuracy, precision, etc.) if labels provided
    
    Returns:
        Dict with health_index, performance metrics and action flags
    """
    try:
        # 1. Calculate health_index via prediction_event_publisher
        # Trigger alerts only if explicitly requested (once at the end)
        health_index = await publicador_predicao.process_prediction_batch(
            model_key=model_key,
            predictions=predictions,
            probabilities=probabilities,
            pseudo_confidences=pseudo_confidences if pseudo_confidences else [],
            is_classification=is_classification,
            input_data=input_data,
            db_session=db,
            trigger_alerts=trigger_alerts
        )
        
        # 2. Calculate confidence metrics (real, not estimated)
        confidences = []
        if is_classification:
            for probs in probabilities:
                if probs and len(probs) > 0:
                    max_prob = max(probs)
                    confidences.append(max_prob)
                    logger.debug(f"Confidence from probabilities: {max_prob:.4f} (probs: {probs})")
        else:
            # Regression: use pseudo_confidences
            if pseudo_confidences:
                confidences = pseudo_confidences
                logger.debug(f"Confidences from pseudo: {confidences}")
        
        confidence_avg = sum(confidences) / len(confidences) if confidences else 0.0
        low_confidence_count = len([c for c in confidences if c < 0.7])
        
        logger.info(
            f"Confidence calculation for {model_key[:8]}: "
            f"avg={confidence_avg:.4f}, low_count={low_confidence_count}, total={len(confidences)}"
        )
        
        # 3. Build performance metrics with all calculated values
        performance_metrics = {
            "confidence_avg": round(confidence_avg, 4),
            "low_confidence_count": low_confidence_count,
            "health_index": round(health_index, 4) if health_index is not None else None
        }
        
        # If real metrics provided (from labels), use them. Otherwise estimate.
        if real_metrics:
            # Use real metrics (from actual labels comparison)
            performance_metrics.update(real_metrics)
        else:
            # Use statistical correlations to approximate metrics without labels
            # For REGRESSION: returns empty dict (cannot estimate MAE/RMSE without labels)
            # For CLASSIFICATION: estimates accuracy, precision, recall, f1, roc_auc
            estimated_metrics = _estimate_traditional_metrics(
                health_index=health_index,
                confidence_avg=confidence_avg,
                low_confidence_count=low_confidence_count,
                total_predictions=len(predictions),
                is_classification=is_classification
            )
            performance_metrics.update(estimated_metrics)
        
        # 4. Record in PerformanceLog (with real or estimated metrics)
        await _record_performance_history(
            db=db,
            model_key=model_key,
            performance_metrics=performance_metrics,
            sample_size=len(predictions),
            health_index=health_index,
            evaluation_type=evaluation_type,
            is_classification=is_classification
        )
        
        # 5. Detect need for action (retraining or switching)
        needs_action = health_index < 0.7  # Alert threshold
        action_recommended = None
        
        if needs_action:
            if health_index < 0.5:
                action_recommended = "URGENT_RETRAINING"  # Critical: urgent retraining
            elif health_index < 0.7:
                action_recommended = "CONSIDER_CHALLENGER_SWITCH"  # Consider switching to challenger
        
        model_type = "Classification" if is_classification else "Regression"
        logger.info(
            f"Health Analysis ({model_type}): {model_key[:8]} | "
            f"Health: {health_index:.4f} | "
            f"Confidence: {confidence_avg:.4f} | "
            f"Action: {action_recommended or 'NONE'}"
        )
        
        return {
            "health_index": health_index,
            "performance_metrics": performance_metrics,
            "needs_action": needs_action,
            "action_recommended": action_recommended
        }
        
    except Exception as e:
        logger.error(f"Error in unified health analysis: {e}", exc_info=True)
        return {
            "health_index": None,
            "performance_metrics": {"confidence_avg": 0.0, "low_confidence_count": 0},
            "needs_action": False,
            "action_recommended": None
        }

async def _calculate_batch_performance_metrics(
    result: Dict[str, Any], 
    original_data: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate real performance metrics based on confidence and data quality.
    This function now returns proxy metrics that will be replaced by the
    health_index from prediction_event_publisher.
    
    DEPRECATED: Use _unified_prediction_health_analysis instead.
    """
    predictions = result.get("predictions", [])
    
    if not predictions:
        return {"confidence_avg": 0.0, "low_confidence_count": 0}
    
    confidences = [
        pred.get("confidence", 0.0) 
        for pred in predictions 
        if pred.get("confidence") is not None
    ]
    
    confidence_avg = sum(confidences) / len(confidences) if confidences else 0.0
    
    return {
        "confidence_avg": round(confidence_avg, 4),
        "low_confidence_count": len([c for c in confidences if c < 0.7])
    }

async def _record_performance_history(
    db: Session,
    model_key: str,
    performance_metrics: Dict[str, float],
    sample_size: int,
    health_index: float = None,
    evaluation_type: str = "prediction",
    is_classification: bool = True
):
    """
    Record performance in PerformanceLog for history.
    Enables temporal analysis and data-driven decision support.
    
    HEALTH INDEX is the PRIMARY metric of the system.
    Traditional metrics are ALWAYS saved:
       - Classification: accuracy, precision, recall, f1_score, roc_auc (estimated or real)
       - Regression: mae, rmse, r2, mse (estimated or real)
    """
    try:
        # Extract auxiliary metrics
        confidence_avg = performance_metrics.get("confidence_avg", 0.0)
        low_conf_count = performance_metrics.get("low_confidence_count", 0)
        data_drift_score = performance_metrics.get("data_drift_risk")
        
        # Use is_classification parameter (already known from caller)
        is_regression = not is_classification
        
        # Build PerformanceLog with appropriate metrics
        log_data = {
            "model_key": model_key,
            "timestamp": datetime.utcnow(),
            "health_index": health_index,
            "evaluation_type": evaluation_type,
            "sample_size": sample_size,
            "data_drift_score": data_drift_score,
            "concept_drift_score": None
        }
        
        # Add classification metrics if present
        if is_classification:
            log_data.update({
                "accuracy": performance_metrics.get("accuracy"),
                "precision": performance_metrics.get("precision"),
                "recall": performance_metrics.get("recall"),
                "f1_score": performance_metrics.get("f1_score"),
                "roc_auc": performance_metrics.get("roc_auc")
            })
        
        # Add regression metrics if present
        if is_regression:
            log_data.update({
                "mae": performance_metrics.get("mae"),
                "rmse": performance_metrics.get("rmse"),
                "r2": performance_metrics.get("r2"),
                "mse": performance_metrics.get("mse")
            })
        
        performance_log = PerformanceLog(**log_data)
        
        db.add(performance_log)
        db.commit()
        
        # Build log message based on model type
        health_str = f"{health_index:.4f}" if health_index is not None else "N/A"
        
        if is_classification:
            acc_str = f"{log_data.get('accuracy'):.4f}" if log_data.get('accuracy') else "N/A"
            prec_str = f"{log_data.get('precision'):.4f}" if log_data.get('precision') else "N/A"
            rec_str = f"{log_data.get('recall'):.4f}" if log_data.get('recall') else "N/A"
            f1_str = f"{log_data.get('f1_score'):.4f}" if log_data.get('f1_score') else "N/A"
            
            logger.info(
                f"Performance recorded (Classification): {model_key[:8]} - "
                f"Health: {health_str} | "
                f"Acc: {acc_str} | Prec: {prec_str} | Rec: {rec_str} | F1: {f1_str} | "
                f"Confidence: {confidence_avg:.4f} ({low_conf_count} low) | "
                f"Type: {evaluation_type}"
            )
        elif is_regression:
            mae_str = f"{log_data.get('mae'):.4f}" if log_data.get('mae') else "N/A"
            rmse_str = f"{log_data.get('rmse'):.4f}" if log_data.get('rmse') else "N/A"
            r2_str = f"{log_data.get('r2'):.4f}" if log_data.get('r2') else "N/A"
            
            logger.info(
                f"Performance recorded (Regression): {model_key[:8]} - "
                f"Health: {health_str} | "
                f"MAE: {mae_str} | RMSE: {rmse_str} | R2: {r2_str} | "
                f"Confidence: {confidence_avg:.4f} ({low_conf_count} low) | "
                f"Type: {evaluation_type}"
            )
        else:
            logger.info(
                f"Performance recorded: {model_key[:8]} - "
                f"Health: {health_str} | "
                f"Confidence: {confidence_avg:.4f} ({low_conf_count} low) | "
                f"Type: {evaluation_type}"
            )
        
    except Exception as e:
        logger.error(f"Error recording performance in history: {e}")
        db.rollback()

