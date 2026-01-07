"""
Model training endpoints router
"""
import logging
import time
from pathlib import Path as PathLib
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, BackgroundTasks, Form, Query, Path
from sqlalchemy.orm import Session

from src.database.config import get_db
from src.schemas.training import (
    RespostaTreinamento, 
    ErroTreinamento,
    ConfiguracaoTreinamento,
    StatusTreinamento,
    RespostaRetreinamento
)
from src.services.file_service import FileService
from src.services.training_service import create_new_models_pipeline
from src.utils.file_utils import gerar_nome_arquivo_seguro

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["Training"])

BASE_FOLDER = PathLib.cwd() / "tmp"
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ACCEPTED_FORMATS = [".csv"]
VALID_METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUC", "R2", "MAE", "MSE", "RMSE"]
VALID_STRATEGIES = ["metric_drop", "periodic", "threshold_based"]

@router.post(
    "/train",
    response_model=RespostaTreinamento,
    summary="Start a new AutoML training pipeline",
    responses={
        400: {"model": ErroTreinamento, "description": "Input data validation error"},
        413: {"model": ErroTreinamento, "description": "File too large"},
        415: {"model": ErroTreinamento, "description": "Unsupported file type"},
        422: {"model": ErroTreinamento, "description": "Invalid file data"},
        500: {"model": ErroTreinamento, "description": "Internal server error"}
    }
)
async def automated_model_training(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    file: UploadFile = File(
        ...,
        description="CSV dataset file containing training data with features and target column. Must include header row with column names and be UTF-8 encoded."
    ),
    target_column: str = Form(
        ...,
        description="Name of the target column in the CSV file that the model should predict. This column contains the dependent variable:\n* **Classification**: Categorical labels (e.g., 'status', 'category', 'class')\n* **Regression**: Continuous numeric values (e.g., 'price', 'temperature', 'quantity')",
        example="target_value"
    ),
    n: int = Form(
        2,
        ge=2,
        le=10,
        description="Number of models to train using different algorithms. Minimum 2 required for Champion/Challenger strategy:\n* **2-10 models**: The best becomes 'Champion' (PRODUCTION), others become 'Challengers'\n* **Recommended**: 3-5 models for optimal performance vs training time tradeoff",
        example=3
    ),
    metric: str = Form(
        "Accuracy",
        description="Optimization metric for model selection and evaluation. Valid options:\n\n**Classification Metrics:**\n* **Accuracy**: Overall correctness (default)\n* **Precision**: Minimize false positives\n* **Recall**: Minimize false negatives\n* **F1**: Balance between precision and recall\n* **AUC**: Area under ROC curve\n\n**Regression Metrics:**\n* **R2**: Coefficient of determination\n* **MAE**: Mean absolute error\n* **MSE**: Mean squared error\n* **RMSE**: Root mean squared error",
        example="Accuracy"
    ),
    evaluation_strategy: str = Form(
        "metric_drop",
        description="Strategy for continuous model performance monitoring. Valid options:\n* **metric_drop**: Trigger alerts when performance drops below threshold (recommended)\n* **periodic**: Scheduled evaluations at fixed intervals\n* **threshold_based**: Alert when absolute metric falls below threshold",
        example="metric_drop"
    ),
    evaluation_interval: int = Form(
        3600,
        ge=300,
        le=86400,
        description="Time interval in seconds between model performance evaluations. Valid range: 300-86400 seconds.\n* **300**: 5 minutes (frequent monitoring)\n* **3600**: 1 hour (recommended default)\n* **86400**: 24 hours (daily checks)",
        example=3600
    ),
    threshold: float = Form(
        0.05,
        ge=0.01,
        le=0.5,
        description="Performance degradation threshold for triggering alerts. Valid range: 0.01-0.5.\n* **0.01**: Very sensitive (1% drop triggers alert)\n* **0.05**: Balanced sensitivity (5% drop, recommended)\n* **0.10**: Less sensitive (10% drop)\n\nExample: threshold=0.05 means alert if accuracy drops from 0.90 to 0.85 or below.",
        example=0.05
    )
):
    """
    ### Description
    Initiates an automated machine learning training pipeline that trains multiple models in parallel 
    and selects the best performer for production deployment using Champion/Challenger strategy.
    
    ### Training Pipeline
    1. Validates CSV file format, size, and data quality
    2. Detects problem type (classification or regression) automatically
    3. Trains N models concurrently using different algorithms (background process)
    4. Evaluates all models and selects best based on optimization metric
    5. Deploys best model as PRODUCTION (Champion), others as CHALLENGER
    6. Configures continuous performance monitoring system
    
    ### CSV Requirements
    - **Format**: CSV with comma separator
    - **Encoding**: UTF-8
    - **Header**: First row with column names (required)
    - **Size**: Maximum 100MB
    - **Records**: Minimum 100 rows recommended
    - **Columns**: Minimum 2 (at least 1 feature + 1 target)
    - **Missing values**: Handled automatically (imputation)
    - **Data types**: Numeric and categorical features supported
    
    ### Problem Type Detection (Automatic)
    - **Classification**: 
      - Categorical target column (string/object type)
      - Numeric target with ≤10 unique values
      - Examples: status (active/inactive), category (A/B/C), quality (low/medium/high)
    - **Regression**: 
      - Numeric target with >10 unique values
      - Examples: price, temperature, quantity, sales volume
    
    ### Algorithms Evaluated
    - **Classification**: Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, LightGBM, XGBoost, Decision Tree, Naive Bayes, K-Neighbors, SVM
    - **Regression**: Linear Regression, Ridge, Lasso, Elastic Net, Random Forest Regressor, Extra Trees Regressor, Gradient Boosting Regressor, LightGBM Regressor, XGBoost Regressor, Decision Tree Regressor
    
    ### Response
    Returns job ID for progress tracking, dataset metadata, and applied configuration.
    Monitor training status via `GET /models/train/{job_id}/status`.
    
    ### Champion/Challenger Strategy
    Best model becomes PRODUCTION Champion, remaining models serve as CHALLENGER alternatives for A/B testing and model governance. Minimum 2 models required.
    """
    
    try:
        validation_errors = []
        
        if not file.filename:
            validation_errors.append("File name is required")
        elif not any(file.filename.lower().endswith(fmt) for fmt in ACCEPTED_FORMATS):
            validation_errors.append(f"Invalid file format. Accepted: {', '.join(ACCEPTED_FORMATS)}")
        
        if metric not in VALID_METRICS:
            validation_errors.append(f"Invalid metric. Accepted: {', '.join(VALID_METRICS)}")
        
        if evaluation_strategy not in VALID_STRATEGIES:
            validation_errors.append(f"Invalid strategy. Accepted: {', '.join(VALID_STRATEGIES)}")
        
        if validation_errors:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid input data",
                    "details": validation_errors,
                    "code": "VALIDATION_ERROR"
                }
            )
        
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Empty file",
                    "details": "The uploaded file contains no data",
                    "code": "EMPTY_FILE"
                }
            )
        
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "File too large",
                    "details": f"Maximum allowed size: {MAX_FILE_SIZE // (1024*1024)}MB",
                    "current_size": f"{len(content) // (1024*1024)}MB",
                    "code": "FILE_TOO_LARGE"
                }
            )
        
        file_service = FileService()
        
        try:
            db_file = file_service.save_uploaded_file(
                content=content,
                filename=file.filename,
                content_type=file.content_type or "text/csv",
                target_column=target_column,
                safe_filename="",
                file_path=PathLib(""),
                db=db
            )
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Invalid dataset",
                    "details": str(e),
                    "code": "INVALID_DATASET"
                }
            )
        
        dataset_rows = getattr(db_file, 'rows', 0)
        dataset_columns = getattr(db_file, 'columns', 0)
        dataset_columns_names = getattr(db_file, 'columns_names', [])
        
        if dataset_rows < 100:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Dataset too small",
                    "details": f"Minimum 100 records, found {dataset_rows}",
                    "code": "INSUFFICIENT_DATA"
                }
            )
        
        if dataset_columns < 2:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Invalid dataset",
                    "details": "At least 2 columns required (1 feature + 1 target)",
                    "code": "INSUFFICIENT_FEATURES"
                }
            )
        
        logger.info(f"Dataset validated: ({dataset_rows}, {dataset_columns}), target={target_column}")
        
        background_tasks.add_task(
            _executar_pipeline_treinamento,
            db_file=db_file,
            target_column=target_column,
            n=n,
            metric=metric,
            evaluation_strategy=evaluation_strategy,
            evaluation_interval=evaluation_interval,
            threshold=threshold
        )
        
        return RespostaTreinamento(
            sucesso=True,
            mensagem=f"Training of {n} model(s) started",
            id_job=db_file.id,
            configuracao=ConfiguracaoTreinamento(
                nome_arquivo=file.filename,
                coluna_alvo=target_column,
                qtd_modelos=n,
                metrica_otimizacao=metric,
                estrategia_monitoramento=evaluation_strategy,
                intervalo_avaliacao=evaluation_interval,
                threshold_degradacao=threshold
            ),
            dataset_info={
                "linhas": dataset_rows,
                "colunas": dataset_columns,
                "tamanho_mb": len(content) / (1024*1024),
                "colunas_disponiveis": dataset_columns_names
            },
            tempo_estimado_minutos=n * 5,
            criado_em=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal training error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "details": "Training request processing failed",
                "code": "INTERNAL_SERVER_ERROR",
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@router.get(
    "/train/{job_id}/status",
    response_model=StatusTreinamento,
    summary="Retrieve training job status"
)
async def get_training_status(
    job_id: int = Path(
        ...,
        description="The unique identifier of the training job returned from the training endpoint.",
        example=42
    ),
    db: Session = Depends(get_db)
):
    """
    ### Query Training Job Progress
    
    Retrieve the current status and details of a training job initiated 
    via the `/models/train` endpoint.
    
    ### Possible Status Values
    - **PENDING:** Waiting for processing to start
    - **RUNNING:** Training in progress
    - **COMPLETED:** Training completed successfully
    - **FAILED:** Training failed due to error
    - **CANCELLED:** Training was cancelled
    
    ### Returns
    - Job ID and current status
    - List of trained models with their metrics
    - Execution time in seconds
    - Creation and update timestamps
    """
    try:
        file_service = FileService()
        file = file_service.get_file_by_id(job_id, db)
        
        from src.repositories.model_repository import ModelRepository
        model_repo = ModelRepository()
        models = model_repo.get_by_file_id(db, job_id)
        
        if models:
            status = "COMPLETED"
            details = f"{len(models)} model(s) trained successfully"
            models_info = [
                {
                    "id": model.id,
                    "nome": model.model,
                    "acuracia": model.accuracy,
                    "estagio": model.stage.value if hasattr(model.stage, 'value') else str(model.stage)
                }
                for model in models
            ]
        else:
            elapsed_time = (datetime.utcnow() - file.created_at).total_seconds()
            if elapsed_time < 600:
                status = "RUNNING"
                details = "Training in progress"
            else:
                status = "FAILED"
                details = "Training failed or was interrupted"
            models_info = []
        
        return StatusTreinamento(
            job_id=job_id,
            status=status,
            detalhes=details,
            modelos_criados=models_info,
            tempo_execucao_segundos=int((datetime.utcnow() - file.created_at).total_seconds()),
            criado_em=file.created_at.isoformat(),
            atualizado_em=datetime.utcnow().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Job not found",
                "details": str(e),
                "code": "JOB_NOT_FOUND"
            }
        )
    except Exception as e:
        logger.error(f"Error querying status: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal error",
                "details": "Failed to query training status",
                "code": "INTERNAL_ERROR"
            }
        )
        
async def _executar_pipeline_treinamento(
    db_file,
    target_column: str, 
    n: int,
    metric: str,
    evaluation_strategy: str,
    evaluation_interval: int,
    threshold: float
):
    """Training pipeline executed in background."""
    from src.database.config import SessionLocal
    
    db = SessionLocal()
    try:
        await create_new_models_pipeline(
            db=db,
            db_file=db_file,
            target_column=target_column,
            n=n,
            metric=metric,
            evaluation_strategy=evaluation_strategy,
            evaluation_interval=evaluation_interval,
            threshold=threshold
        )
        
        logger.info(f"Training pipeline completed for file {db_file.id}")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
    finally:
        db.close()