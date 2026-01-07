"""
Router for human actions based on alerts and analysis.
"""
import logging
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path, File, UploadFile, Form
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from src.database.config import get_db
from src.database.models.Alert import Alert, AlertStatus, AlertType
from src.database.models.Result import Result as DBResult, ModelStage
from src.database.models.File import File as DBFile
from src.repositories.model_repository import ModelRepository
from src.services.alert_service import alert_service
from src.services.file_service import FileService
from src.services.training_service import create_new_models_pipeline
from src.utils.automatic_retraining import AutomaticRetraining
from src.schemas.training import RespostaRetreinamento

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/human-actions", tags=["Human Actions and Governance"])

# Constantes para validação
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ACCEPTED_FORMATS = [".csv"]
VALID_METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUC", "R2", "MAE", "MSE", "RMSE"]

@router.get("/alerts", summary="List alerts requiring human decision and analysis")
async def list_alerts_for_decision(
    status: Optional[str] = Query(
        "OPEN",
        description="Filter alerts by status. Valid options:\n* **OPEN**: New alerts awaiting acknowledgment\n* **ACKNOWLEDGED**: Alerts reviewed but not yet resolved\n* **RESOLVED**: Completed alerts with actions taken\n* **CLOSED**: Archived alerts no longer requiring attention",
        example="OPEN"
    ),
    limit: int = Query(
        50,
        ge=1,
        le=200,
        description="Maximum number of alerts to return per request. Valid range: 1 to 200",
        example=50
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Number of alerts to skip for pagination (0-indexed)",
        example=0
    ),
    model_key: Optional[str] = Query(
        None,
        description="Filter alerts by specific model unique identifier. If not provided, returns alerts for all models",
        example="a66ccf8aa3634dfb5afe46cb19f79d01"
    ),
    db: Session = Depends(get_db)
):
    """
    ### Description
    Retrieves alerts requiring human decision-making and governance actions for ML operations.
    Provides alert metadata, model context, and recommended actions for performance degradation, 
    data drift, and anomaly detection.
    
    ### Response Structure
    Returns paginated alert list with metadata including alert ID, associated model, alert type, 
    status, timestamps, and recommended governance actions. Includes total count and pagination parameters.
    """
    try:
        status_mapping = {
            "OPEN": AlertStatus.open,
            "ACKNOWLEDGED": AlertStatus.acknowledged,
            "RESOLVED": AlertStatus.resolved,
            "CLOSED": AlertStatus.closed
        }
        
        status_filter = status_mapping.get(status.upper(), AlertStatus.open)
        
        query = db.query(Alert).filter(Alert.status == status_filter)
        
        if model_key:
            query = query.filter(Alert.model_key == model_key)
        
        alerts = query.order_by(Alert.created_at.desc()).offset(offset).limit(limit).all()
        
        alerts_with_model_info = []
        for alert in alerts:
            model_info = db.query(DBResult).filter(DBResult.key == alert.model_key).first()
            
            alert_data = {
                "alert_id": alert.id,
                "model_key": alert.model_key,
                "model_name": model_info.model if model_info else "Unknown",
                "model_stage": model_info.stage.value if model_info and model_info.stage else "Unknown",
                "alert_type": alert.alert_type.value,
                "status": alert.status.value,
                "details": alert.details,
                "created_at": alert.created_at.isoformat(),
                "updated_at": alert.updated_at.isoformat(),
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                "recommended_actions": _get_recommended_actions(alert.alert_type)
            }
            
            alerts_with_model_info.append(alert_data)
        
        # Get alert summary (with error handling)
        summary = await _get_alerts_summary(db)
        
        return {
            "alerts": alerts_with_model_info,
            "total_shown": len(alerts_with_model_info),
            "status_filter": status,
            "model_filter": model_key,
            "pagination": {
                "limit": limit,
                "offset": offset
            },
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error listing alerts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching alerts: {str(e)}")


@router.post("/alerts/{alert_id}/approve-retraining", 
             response_model=RespostaRetreinamento,
             summary="Approve and start complete retraining with new data")
async def approve_retraining(
    alert_id: int = Path(
        ...,
        description="Unique identifier of the alert that triggered the retraining decision",
        example=123
    ),
    background_tasks: BackgroundTasks = None,
    file: UploadFile = File(
        ...,
        description="New CSV training dataset with updated data to replace degraded model. Must be UTF-8 encoded with header row."
    ),
    target_column: str = Form(
        ...,
        description="Name of the target column in the CSV file that the model should predict",
        example="label"
    ),
    n: int = Form(
        3,
        ge=2,
        le=10,
        description="Number of new models to train. Minimum 2 required for Champion/Challenger strategy. Range: 2-10",
        example=3
    ),
    metric: str = Form(
        "Accuracy",
        description="Optimization metric for model selection. Valid options:\n* **Accuracy**: Overall correctness\n* **Precision**: Minimize false positives\n* **Recall**: Minimize false negatives\n* **F1**: Balance precision/recall\n* **AUC**: ROC curve area",
        example="Accuracy"
    ),
    evaluation_strategy: str = Form(
        "metric_drop",
        description="Monitoring strategy for performance tracking. Valid options:\n* **metric_drop**: Alert on performance drops\n* **periodic**: Scheduled checks\n* **threshold_based**: Absolute threshold alerts",
        example="metric_drop"
    ),
    evaluation_interval: int = Form(
        3600,
        ge=300,
        le=86400,
        description="Time interval in seconds between performance evaluations. Range: 300-86400 (5 min - 24 hours)",
        example=3600
    ),
    threshold: float = Form(
        0.05,
        ge=0.01,
        le=0.5,
        description="Performance degradation threshold for alerts. Range: 0.01-0.5 (1%-50% drop)",
        example=0.05
    ),
    model_types: Optional[str] = Form(
        None,
        description="Comma-separated list of model types to train (PyCaret codes). Examples:\n"
                    "* Classification: 'rf,et,lightgbm,gbc,ada,ridge,dt,knn,nb,svm'\n"
                    "* Regression: 'rf,et,lightgbm,gbr,ada,ridge,dt,knn,svr'\n"
                    "If not specified, uses default models: rf,et,lightgbm,gbc (classification) or rf,et,lightgbm,gbr (regression)",
        example="rf,et,lightgbm,gbc"
    ),
    db: Session = Depends(get_db)
):
    """
    **Complete retraining flow from alert with automatic model replacement**
    
    This endpoint implements the full Human-in-the-Loop retraining workflow:
    
    **Flow:**
    1. Human receives degradation alert for a model in production
    2. Human analyzes the alert and decides to retrain with new data
    3. Human uploads new CSV file through this endpoint
    4. System initiates AutoML training job (background)
    5. After training completes, system automatically:
       - Identifies best new model (new champion)
       - Archives ALL old models from the alert lineage
       - Promotes new model to champion
       - Resolves the alert
    
    **Input Requirements:**
    - `alert_id`: The ID of the alert justifying retraining
    - `file`: New CSV file with updated/fresh training data
    - `target_column`: Target variable name
    - `n`: Number of models to train (default: 3)
    - `metric`: Optimization metric (Accuracy, F1, etc.)
    
    **CSV Requirements:**
    - Format: CSV with comma separator
    - Encoding: UTF-8
    - Size: Maximum 100MB
    - Minimum 100 rows
    - At least 2 columns (1 feature + target)
    
    **Returns:**
    - `new_job_id`: ID of the retraining job for tracking
    - `alert_resolved_id`: The alert that will be resolved
    - `old_model_id`: Key of the model being replaced
    
    **Post-Training (Automatic):**
    After training completes in background, the system will:
    - Archive all models from the old lineage (same file_id)
    - Promote the best new model to champion
    - Mark alert as resolved with details
    
    **Example Response:**
    ```json
    {
        "message": "Retraining job initiated successfully.",
        "new_job_id": 42,
        "alert_resolved_id": 123,
        "old_model_id": "abc123...",
        "old_model_name": "RandomForestClassifier",
        "timestamp": "2025-11-01T10:30:00"
    }
    ```
    """
    try:
        # 1. Validar o alerta
        alert = db.query(Alert).filter(Alert.id == alert_id).first()
        if not alert:
            raise HTTPException(
                status_code=404,
                detail=f"Alert {alert_id} not found."
            )
        
        if alert.status == AlertStatus.resolved:
            raise HTTPException(
                status_code=400,
                detail=f"Alert {alert_id} is already resolved."
            )
        
        # 2. Obter informações do modelo antigo
        old_model = db.query(DBResult).filter(DBResult.key == alert.model_key).first()
        if not old_model:
            raise HTTPException(
                status_code=404,
                detail=f"Model {alert.model_key} associated with alert not found."
            )
        
        # 3. Validar arquivo
        validation_errors = []
        
        if not file.filename:
            validation_errors.append("File name is required")
        elif not any(file.filename.lower().endswith(fmt) for fmt in ACCEPTED_FORMATS):
            validation_errors.append(f"Invalid file format. Accepted: {', '.join(ACCEPTED_FORMATS)}")
        
        if metric not in VALID_METRICS:
            validation_errors.append(f"Invalid metric. Accepted: {', '.join(VALID_METRICS)}")
        
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid input data",
                    "details": validation_errors
                }
            )
        
        # 4. Ler e validar conteúdo do arquivo
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # 5. Salvar arquivo e criar job
        file_service = FileService()
        
        try:
            db_file = file_service.save_uploaded_file(
                content=content,
                filename=f"retraining_alert_{alert_id}_{file.filename}",
                content_type=file.content_type or "text/csv",
                target_column=target_column,
                safe_filename="",
                file_path="",
                db=db,
                optimization_metric=metric
            )
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Invalid dataset",
                    "details": str(e)
                }
            )
        
        # 6. Vincular job ao alerta
        alert.retraining_job_id = db_file.id
        alert.status = AlertStatus.acknowledged
        alert.details += (
            f"\n\n[RETRAINING INITIATED] Human approved retraining at {datetime.utcnow().isoformat()}.\n"
            f"New data file: {file.filename}\n"
            f"Target column: {target_column}\n"
            f"Models to train: {n}\n"
            f"Job ID: {db_file.id}\n"
            f"Status: Training in progress..."
        )
        alert.updated_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Retraining initiated for alert {alert_id}, job {db_file.id}")
        
        # 7. Processar model_types
        model_types_list = None
        if model_types:
            model_types_list = [m.strip() for m in model_types.split(',') if m.strip()]
            logger.info(f"User specified model types: {model_types_list}")
        
        # 8. Iniciar pipeline de treinamento em background
        background_tasks.add_task(
            _executar_pipeline_retrainamento,
            db_file=db_file,
            target_column=target_column,
            n=n,
            metric=metric,
            evaluation_strategy=evaluation_strategy,
            evaluation_interval=evaluation_interval,
            threshold=threshold,
            alert_id=alert_id,
            model_types=model_types_list
        )
        
        # 9. Retornar resposta
        return RespostaRetreinamento(
            message="Retraining job initiated successfully. Models will be automatically replaced after training.",
            new_job_id=db_file.id,
            alert_resolved_id=alert_id,
            old_model_id=old_model.key,
            old_model_name=old_model.model,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating retraining from alert: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate retraining: {str(e)}"
        )

@router.post("/alerts/{alert_id}/promote-challenger", summary="Promote best challenger to production based on alert")
async def promote_challenger_via_alert(
    alert_id: int = Path(
        ...,
        description="Unique identifier of the alert justifying the challenger promotion decision",
        example=42
    ),
    db: Session = Depends(get_db)
):
    """
    ### Description
    Promotes the best-performing challenger model to production champion status based on governance alert analysis.
    
    This endpoint enables human-in-the-loop decision-making when alerts indicate that a challenger model
    outperforms the current production champion. The promotion process includes automatic champion demotion
    and alert resolution.
    
    ### Path Parameters
    - **alert_id**: Unique identifier of the alert justifying the promotion
    
    ### Promotion Logic
    1. Validate alert exists and retrieve associated model lineage
    2. Query all CHALLENGER stage models from the same dataset/lineage
    3. Identify best challenger based on optimization metric (accuracy, F1, etc.)
    4. Demote current PRODUCTION model to CHALLENGER stage
    5. Promote best challenger to PRODUCTION stage
    6. Update alert status to RESOLVED
    7. Log governance action for audit trail
    
    ### Response Fields
    - **message**: Confirmation message of successful promotion
    - **old_champion**: Key and name of demoted production model
    - **new_champion**: Key and name of promoted challenger model
    - **alert_resolved_id**: ID of the resolved alert
    - **timestamp**: ISO 8601 timestamp of promotion execution
    
    ### Use Cases
    - Performance degradation alert indicates challenger outperforms champion
    - A/B testing results favor challenger over production model
    - Governance decision to replace underperforming champion
    """
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")

    model_in_alert = db.query(DBResult).filter(DBResult.key == alert.model_key).first()
    if not model_in_alert:
        raise HTTPException(status_code=404, detail=f"Model {alert.model_key} associated with the alert not found.")

    model_repo = ModelRepository()
    
    # 1. Buscar champion atual ANTES da promoção
    from src.database.models.Result import ModelStage
    old_champion = db.query(DBResult).filter(
        DBResult.file_id == model_in_alert.file_id,
        DBResult.stage == ModelStage.champion,
        DBResult.is_active == True
    ).first()
    
    # 2. Buscar a métrica de otimização usada no treinamento
    from src.database.models.File import File as DBFile
    file_record = db.query(DBFile).filter(DBFile.id == model_in_alert.file_id).first()
    optimization_metric = file_record.optimization_metric if file_record else "F1"
    
    logger.info(f"Switching champion using metric: {optimization_metric}")
    
    # 3. Get all challengers (WITHOUT pre-sorting by DB metrics - they may be outdated!)
    all_challengers = db.query(DBResult).filter(
        DBResult.file_id == model_in_alert.file_id,
        DBResult.stage == ModelStage.challenger,
        DBResult.is_active == True
    ).all()
    
    if not all_challengers:
        raise HTTPException(
            status_code=404, 
            detail=f"No active challenger found for the lineage of model {model_in_alert.key[:8]}."
        )
    
    # 4. Get recent performance logs to check current health of challengers
    from src.database.models.PerformanceLog import PerformanceLog
    
    # Get latest performance evaluation for each challenger
    logger.info(f"Evaluating {len(all_challengers)} challengers based on recent performance logs...")
    
    challenger_performances = []
    
    for challenger in all_challengers:
        # Get most recent performance log for this challenger
        latest_perf = db.query(PerformanceLog).filter(
            PerformanceLog.model_key == challenger.key
        ).order_by(desc(PerformanceLog.timestamp)).first()
        
        if latest_perf and latest_perf.health_index is not None:
            # Use recent health index to select best challenger
            # Map optimization metric to performance log field
            metric_map = {
                "Accuracy": latest_perf.accuracy,
                "F1": latest_perf.f1_score,
                "Precision": latest_perf.precision,
                "Recall": latest_perf.recall,
                "AUC": latest_perf.roc_auc
            }
            
            recent_metric = metric_map.get(optimization_metric, latest_perf.f1_score)
            
            # Use recent metric if available, otherwise fall back to stored
            if recent_metric is not None and recent_metric > 0:
                metric_to_use = recent_metric
                source = "RECENT_PERFORMANCE"
            else:
                metric_to_use = getattr(challenger, optimization_metric.lower().replace("auc", "roc_auc"), 0.0)
                source = "STORED_METRICS"
            
            challenger_performances.append({
                "model": challenger,
                "metric_value": metric_to_use,
                "health_index": latest_perf.health_index,
                "source": source,
                "timestamp": latest_perf.timestamp
            })
            
            logger.info(
                f"  → {challenger.model} ({challenger.key[:8]}): "
                f"{optimization_metric}={metric_to_use:.4f} "
                f"(Health={latest_perf.health_index:.3f}, Source={source})"
            )
        else:
            # No recent performance data, use stored metrics
            metric_to_use = getattr(challenger, optimization_metric.lower().replace("auc", "roc_auc"), 0.0)
            
            challenger_performances.append({
                "model": challenger,
                "metric_value": metric_to_use,
                "health_index": None,
                "source": "STORED_METRICS",
                "timestamp": None
            })
            
            logger.warning(
                f"  → {challenger.model} ({challenger.key[:8]}): "
                f"{optimization_metric}={metric_to_use:.4f} "
                f"(No recent performance data, using stored metrics)"
            )
    
    # Sort by metric value (best first)
    challenger_performances.sort(key=lambda x: x["metric_value"], reverse=True)
    
    best_challenger = challenger_performances[0]["model"]
    metric_value = challenger_performances[0]["metric_value"]
    total_challengers = len(all_challengers)
    metrics_source = challenger_performances[0]["source"]
    
    logger.info(f"✅ Best challenger selected: {best_challenger.model} ({optimization_metric}={metric_value:.4f}, Source={metrics_source})")
    
    # 3. Promote best challenger and demote old champion
    success = model_repo.promote_to_champion(db, best_challenger.key)
    
    if not success:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to promote challenger {best_challenger.key[:8]} to champion."
        )
    
    # 5. Update alert as resolved with complete details
    resolution_details = (
        f"\n\nRESOLUTION:\n"
        f"  - Best Challenger promoted: {best_challenger.model} ({optimization_metric}: {metric_value:.4f})\n"
        f"  - Old Champion demoted: {old_champion.model if old_champion else 'N/A'} -> Challenger stage\n"
        f"  - Total challengers evaluated: {total_challengers}\n"
        f"  - Selection criteria: {optimization_metric} (metric used in training)\n"
        f"  - Metrics source: {metrics_source}\n"
        f"  - Timestamp: {datetime.utcnow().isoformat()}"
    )
    
    alert.status = AlertStatus.resolved
    alert.details += resolution_details
    alert.resolved_at = datetime.utcnow()
    alert.updated_at = datetime.utcnow()
    db.commit()
    
    return {
        "message": f"Best challenger promoted to champion based on {optimization_metric}.",
        "new_champion": {
            "key": best_challenger.key,
            "model": best_challenger.model,
            "optimization_metric": optimization_metric,
            "metric_value": metric_value,
            "f1_score": best_challenger.f1_score,
            "accuracy": best_challenger.accuracy,
            "metrics_source": metrics_source
        },
        "old_champion": {
            "key": old_champion.key if old_champion else None,
            "model": old_champion.model if old_champion else None,
            "new_stage": "challenger"
        },
        "total_challengers_evaluated": total_challengers,
        "alert_id": alert_id,
        "status": "resolved",
        "timestamp": datetime.utcnow().isoformat()
    }


async def _executar_pipeline_retrainamento(
    db_file: DBFile,
    target_column: str,
    n: int,
    metric: str,
    evaluation_strategy: str,
    evaluation_interval: int,
    threshold: float,
    alert_id: int,
    model_types: list = None
):
    """
    Execute retraining pipeline in background.
    This includes the automatic replacement logic.
    """
    from src.database.config import SessionLocal
    
    db = SessionLocal()
    try:
        logger.info(f"Starting retraining pipeline for alert {alert_id}, job {db_file.id}")
        if model_types:
            logger.info(f"User-specified model types: {model_types}")
        
        # Chama o pipeline com alert_id para ativar o fluxo de substituição automática
        await create_new_models_pipeline(
            db=db,
            db_file=db_file,
            target_column=target_column,
            n=n,
            metric=metric,
            evaluation_strategy=evaluation_strategy,
            evaluation_interval=evaluation_interval,
            threshold=threshold,
            alert_id=alert_id,  # CRUCIAL: Ativa o fluxo de substituição
            model_types=model_types  # Tipos específicos de modelos
        )
        
        logger.info(f"Retraining pipeline completed for alert {alert_id}")
        
    except Exception as e:
        logger.error(f"Error in retraining pipeline for alert {alert_id}: {e}", exc_info=True)
        
        # Atualizar alerta com erro
        try:
            alert = db.query(Alert).filter(Alert.id == alert_id).first()
            if alert:
                alert.details += f"\n\n[ERROR] Retraining failed: {str(e)}"
                alert.status = AlertStatus.open  # Voltar para aberto
                db.commit()
        except Exception as update_error:
            logger.error(f"Failed to update alert status: {update_error}")
            
    finally:
        db.close()


def _get_recommended_actions(alert_type):
    """Return recommended actions based on alert type."""
    recommendations = {
        AlertType.performance_degradation: [
            "Analyze model performance history",
            "Complete Retraining with new data (POST /alerts/{id}/approve-retraining)",
            "Promote best Challenger (POST /alerts/{id}/promote-challenger)", 
            "Verify input data quality",
            "Check for data drift or distribution changes"
        ],
        AlertType.challenger_available: [
            "Review challenger model performance metrics",
            "Compare challenger vs champion accuracy",
            "Approve promotion (POST /alerts/{id}/promote-challenger)",
            "Monitor both models before decision"
        ]
    }
    
    return recommendations.get(alert_type, ["Analyze alert details and model context"])

async def _get_alerts_summary(db: Session):
    """Get alerts summary for dashboard."""
    return alert_service.get_alert_summary(db)