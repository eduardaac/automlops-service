"""
Training Service
"""
import logging
import time
from datetime import datetime
from sqlalchemy.orm import Session
from pathlib import Path

from src.database.models.File import File as DBFile
from src.database.models.Result import Result as DBResult, ModelStage
from src.database.models.PerformanceLog import PerformanceLog
from src.repositories.model_repository import ModelRepository
from src.utils.automl_handler import automl_handler
from src.streaming.kafka_handler import kafka_handler
from src.middleware.metrics_middleware import track_training_job, update_model_metrics

logger = logging.getLogger(__name__)

async def create_new_models_pipeline(
    db: Session,
    db_file: DBFile,
    target_column: str,
    n: int = 1,
    metric: str = "Accuracy",
    evaluation_strategy: str = "metric_drop",
    evaluation_interval: int = 3600,
    threshold: float = 0.05,
    alert_id: int = None,
    model_types: list = None
):
    """
    Complete pipeline for creating trained models.

    Args:
        db: Database session.
        db_file: Data file for training.
        target_column: Target column name.
        n: Number of models to train.
        metric: Evaluation metric.
        evaluation_strategy: Evaluation strategy.
        evaluation_interval: Evaluation interval.
        threshold: Alert threshold.
        alert_id: Optional alert ID for retraining flow.
        model_types: Optional list of specific model types to train (PyCaret codes).

    Returns:
        Dictionary with training results.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Starting training pipeline: {n} models for {db_file.name}")
        
        track_training_job("started")
        
        models_to_train = max(n, 2) if n == 1 else n
        
        model_results = await automl_handler.train_models(
            file_path=db_file.content_path,
            target_column=target_column,
            n=models_to_train,
            metric=metric,
            model_types=model_types,
            diversification=True,
            evaluation_strategy=evaluation_strategy,
            evaluation_interval=evaluation_interval,
            threshold=threshold
        )
        
        if not model_results:
            track_training_job("failed")
            raise ValueError("No models trained")
        
        # IMPORTANT: If retraining, archive old models BEFORE assigning new stages
        if alert_id:
            from src.repositories.model_repository import ModelRepository
            model_repo = ModelRepository()
            
            # Find old models from the same file_id and archive them
            old_models = model_repo.get_by_file_id(db, db_file.id)
            if old_models:
                logger.info(f"Archiving {len(old_models)} old models before creating new lineage...")
                for old_model in old_models:
                    old_model.stage = ModelStage.archived
                    old_model.is_active = False
                db.commit()
                logger.info(f"Old models archived successfully")
        
        await _assign_model_stages(db, model_results, db_file, n)
        
        duration = time.time() - start_time
        track_training_job("completed")
        
        logger.info(f"Pipeline completed: {len(model_results)} models created in {duration:.2f}s")
        
        # Se este é um retreinamento a partir de um alerta, processar a substituição
        if alert_id:
            from src.services.retraining_service import retraining_service
            success = await retraining_service.complete_retraining_job(
                db=db,
                job_id=db_file.id,
                alert_id=alert_id
            )
            if success:
                logger.info(f"Retraining flow completed for alert {alert_id}")
            else:
                logger.warning(f"Retraining flow failed for alert {alert_id}")
        
        return {
            "models": model_results,
            "total_trained": len(model_results),
            "duration_seconds": duration
        }
        
    except Exception as e:
        track_training_job("failed")
        logger.error(f"Training failure: {e}")
        raise

async def _assign_model_stages(
    db: Session, 
    model_results: list, 
    db_file: DBFile, 
    n_requested: int
):
    """
    Assign stages to trained models with consistency checks.
    
    Stage assignment logic:
    - Position 0: CHAMPION (best model)
    - Positions 1 to n_requested-1: CHALLENGER
    - Positions n_requested+: ARCHIVED (extras beyond requested count)

    Args:
        db: Database session.
        model_results: List of trained models.
        db_file: Associated file.
        n_requested: Number of models requested by user (e.g., 3).
    """
    sorted_models = sorted(model_results, key=lambda x: x.get('accuracy', 0), reverse=True)
    
    model_repo = ModelRepository()
    
    logger.info(f"ASSIGNING STAGES:")
    logger.info(f"  - Total models trained: {len(sorted_models)}")
    logger.info(f"  - Models requested (n): {n_requested}")
    logger.info(f"  - Models to keep active: {n_requested} (1 champion + {n_requested-1} challengers)")
    
    for i, model_info in enumerate(sorted_models):
        if i == 0:
            stage = ModelStage.champion
        elif i < n_requested:  # ✅ CORRIGIDO: Usa n_requested em vez de hardcoded 3
            stage = ModelStage.challenger
        else:
            stage = ModelStage.archived
        
        is_active = stage != ModelStage.archived
        
        logger.info(
            f"  - Model {i+1}: {model_info.get('model')} "
            f"-> stage={stage.value}, is_active={is_active} "
            f"(accuracy: {model_info.get('accuracy', 0):.4f})"
        )
        
        if not model_info.get('accuracy') or model_info['accuracy'] < 0 or model_info['accuracy'] > 1:
            logger.warning(f"Invalid accuracy for model {model_info.get('key', 'unknown')}: {model_info.get('accuracy')}")
            continue
        
        model_data = {
            'key': model_info['key'],
            'model': model_info['model'],
            'accuracy': model_info['accuracy'],
            'precision': model_info.get('precision'),
            'recall': model_info.get('recall'),
            'f1_score': model_info.get('f1_score'),
            'roc_auc': model_info.get('roc_auc'),
            'cross_val_roc_auc': model_info.get('cross_val_roc_auc'),
            # Regression metrics
            'mae': model_info.get('mae'),
            'rmse': model_info.get('rmse'),
            'r2': model_info.get('r2'),
            'mse': model_info.get('mse'),
            'file_id': db_file.id,
            'pickle_path': model_info['pickle_path'],
            'stage': stage,
            'is_active': is_active,
            'evaluation_strategy': model_info.get('evaluation_strategy', 'metric_drop'),
            'evaluation_interval': model_info.get('evaluation_interval', 3600),
            'threshold': model_info.get('threshold', 0.05),
            'created_at': datetime.utcnow(),
            'last_evaluated': datetime.utcnow()
        }
        
        db_model = model_repo.create(db, model_data)
        
        await _create_initial_performance_log(db, db_model)
        
        update_model_metrics(
            model_id=db_model.key[:8],
            model_name=db_model.model,
            model_type=db_model.model,
            stage=db_model.stage.value,
            metrics={
                'accuracy': db_model.accuracy or 0.0,
                'precision': db_model.precision or 0.0,
                'recall': db_model.recall or 0.0,
                'f1_score': db_model.f1_score or 0.0
            }
        )
        
        await _publish_training_event([db_model])

async def _publish_training_event(models: list):
    """
    Publish training event to Kafka.

    Args:
        models: List of trained models.
    """
    try:
        for model in models:
            await kafka_handler.publish_retraining_event(
                model_key=model.key,
                event_type="model_trained",
                new_metrics={
                    "accuracy": model.accuracy,
                    "precision": model.precision,
                    "recall": model.recall,
                    "f1_score": model.f1_score,
                    "roc_auc": model.roc_auc,
                    "stage": model.stage.value
                }
            )
    except Exception as e:
        logger.error(f"Event publishing error: {e}")

async def _create_initial_performance_log(db: Session, model: DBResult):
    """
    Create initial performance log.

    Args:
        db: Database session.
        model: Trained model.
    """
    try:
        performance_log = PerformanceLog(
            model_key=model.key,
            timestamp=datetime.utcnow(),
            accuracy=model.accuracy,
            precision=model.precision,
            recall=model.recall,
            f1_score=model.f1_score,
            roc_auc=model.roc_auc,
            evaluation_type="initial_training",
            sample_size=None
        )
        
        db.add(performance_log)
        db.commit()
        
    except Exception as e:
        logger.error(f"Performance log creation error: {e}")
        db.rollback()