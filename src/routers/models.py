"""
Models management router
"""
import logging
from fastapi import APIRouter, Depends, HTTPException, Path, File, UploadFile, Form, BackgroundTasks, Query
from sqlalchemy.orm import Session

from src.database.config import get_db
from src.schemas.training import ListaModelos, RespostaPromocao
from src.services.model_service import ModelService
from src.database.models.Result import Result as DBResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/models", tags=["Model Management"])

@router.get("/", response_model=ListaModelos, summary="List all models with optional filtering")
async def list_models(
    active: bool = Query(
        None,
        description="Filter models by activation status. Valid options:\n* **true**: Show only active models (currently in use)\n* **false**: Show only inactive/archived models\n* **null**: Show all models regardless of status",
        example=True
    ),
    db: Session = Depends(get_db)
):
    """
    ### Description
    Retrieves all trained models with optional filtering by activation status.
    Supports both classification and regression models.
    
    ### Query Parameters
    - **active=true**: Only active models (available for predictions)
    - **active=false**: Only inactive/archived models
    - **active=null**: All models (default)
    
    ### Response
    Returns model array with:
    - **Classification models**: accuracy, precision, recall, f1_score, roc_auc
    - **Regression models**: mae, rmse, r², mse
    - **Common fields**: model name, stage (champion/challenger/archived), activation status, timestamps
    
    ### Model Stages
    - **champion**: Production model (active predictions)
    - **challenger**: Alternative model (A/B testing)
    - **archived**: Inactive model (historical record)
    """
    query = db.query(DBResult)
    if active is not None:
        query = query.filter(DBResult.is_active == active)
    
    models = query.all()
    
    result_list = []
    for model in models:
        # Detectar tipo de modelo baseado em métricas primárias
        # Se MAE/RMSE existem e são > 0, é regressão
        is_regression = (
            (model.mae is not None and model.mae > 0) or 
            (model.rmse is not None and model.rmse > 0)
        )
        
        model_data = {
            "id": model.id,
            "chave": model.key,
            "nome_modelo": model.model,
            "estagio": model.stage.value,
            "ativo": model.is_active,
            "criado_em": model.created_at.isoformat(),
            "arquivo_dataset": model.file.name if model.file else None
        }
        
        # Se é modelo de regressão - mostrar apenas métricas de regressão
        if is_regression:
            model_data.update({
                "mae": model.mae,
                "rmse": model.rmse,
                "r2": model.r2,
                "mse": model.mse,
                "acuracia": None,
                "precisao": None,
                "recall": None,
                "f1_score": None,
                "roc_auc": None
            })
        # Se é modelo de classificação - mostrar apenas métricas de classificação
        else:
            model_data.update({
                "acuracia": model.accuracy,
                "precisao": model.precision,
                "recall": model.recall,
                "f1_score": model.f1_score,
                "roc_auc": model.roc_auc,
                "mae": None,
                "rmse": None,
                "r2": None,
                "mse": None
            })
        
        result_list.append(model_data)
    
    return {
        "total": len(models),
        "modelos": result_list
    }

@router.post(
    "/{model_key}/promote",
    response_model=RespostaPromocao,
    summary="Promote challenger model to production"
)
async def promote_model(
    model_key: str = Path(
        ...,
        description="Unique identifier of the challenger model to promote to production stage",
        example="a66ccf8aa3634dfb5afe46cb19f79d01"
    ),
    db: Session = Depends(get_db)
):
    """
    ### Description
    Promotes a challenger model to production (champion) status and demotes the current champion to challenger.
    Implements Champion/Challenger pattern for model governance.
    
    ### Requirements
    - Model must exist and be active
    - Model must have CHALLENGER stage
    - System automatically demotes current champion
    
    ### Effects
    1. Target model promoted to CHAMPION (production)
    2. Previous champion demoted to CHALLENGER
    3. System maintains exactly ONE champion at all times
    
    ### Use Cases
    - Promote better-performing challenger after A/B testing
    - Model replacement after validation
    - Recovery from production model degradation
    
    ### Response
    Confirmation with promoted model key and previous champion information.
    """
    try:
        model_service = ModelService()
        return model_service.promote_model(model_key, db)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        raise HTTPException(status_code=500, detail="Internal error")

@router.post("/{model_key}/archive", summary="Archive and deactivate a model")
async def archive_model(
    model_key: str = Path(
        ...,
        description="Unique identifier of the model to archive and deactivate",
        example="a66ccf8aa3634dfb5afe46cb19f79d01"
    ),
    db: Session = Depends(get_db)
):
    """
    ### Description
    Archives and deactivates a model while preserving historical records for audit and compliance.
    
    ### Effects
    1. Model stage set to ARCHIVED
    2. is_active flag set to false
    3. Model removed from prediction endpoints
    4. Historical data preserved (not deleted)
    
    ### Use Cases
    - Remove underperforming challengers
    - Clean up old model versions
    - Compliance with data retention policies
    
    ### Note
    Cannot archive the current champion (production) model. Promote another model first.
    """
    try:
        model_service = ModelService()
        success = model_service.archive_model(model_key, db)
        
        if not success:
            raise HTTPException(status_code=404, detail="Modelo não encontrado")
        
        return {"mensagem": "Modelo arquivado com sucesso"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao arquivar modelo: {e}")
        raise HTTPException(status_code=500, detail="Erro interno")

@router.get("/{model_key}", summary="Get detailed information about a specific model")
async def get_model_details(
    model_key: str = Path(
        ...,
        description="Unique identifier of the model to retrieve detailed information",
        example="a66ccf8aa3634dfb5afe46cb19f79d01"
    ),
    db: Session = Depends(get_db)
):
    """
    ### Description
    Retrieves comprehensive metadata for a specific model by unique key.
    Returns different metrics based on model type (classification vs regression).
    
    ### Response Fields
    **Common to all models:**
    - model_key, model_name, stage, active status, timestamps
    - dataset information, training configuration
    
    **Classification models:**
    - accuracy, precision, recall, f1_score, roc_auc
    
    **Regression models:**
    - mae, rmse, r², mse
    
    ### Use Cases
    - Model inspection before promotion
    - Performance comparison
    - Audit and compliance reporting
    """
    try:
        model_service = ModelService()
        model = model_service.get_model_details(model_key, db)
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Detectar tipo de modelo baseado em métricas primárias
        # Se MAE/RMSE existem e são > 0, é regressão
        is_regression = (
            (model.mae is not None and model.mae > 0) or 
            (model.rmse is not None and model.rmse > 0)
        )
        
        base_response = {
            "chave": model.key,
            "nome": model.model,
            "estagio": model.stage.value,
            "ativo": model.is_active,
            "criado_em": model.created_at.isoformat(),
            "ultima_avaliacao": model.last_evaluated.isoformat() if model.last_evaluated else None
        }
        
        if is_regression:
            # Modelo de regressão - mostrar apenas métricas de regressão
            return {
                **base_response,
                "mae": model.mae,
                "rmse": model.rmse,
                "r2": model.r2,
                "mse": model.mse,
                "acuracia": None,
                "precisao": None,
                "recall": None,
                "f1_score": None,
                "roc_auc": None
            }
        else:
            # Modelo de classificação - mostrar apenas métricas de classificação
            return {
                **base_response,
                "acuracia": model.accuracy,
                "precisao": model.precision,
                "recall": model.recall,
                "f1_score": model.f1_score,
                "roc_auc": model.roc_auc,
                "mae": None,
                "rmse": None,
                "r2": None,
                "mse": None
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model details: {e}")
        raise HTTPException(status_code=500, detail="Internal error")
@router.post("/retrain-lineage/{lineage_model_key}", summary="Retrain entire model lineage with new dataset")
async def retrain_lineage_with_new_data(
    background_tasks: BackgroundTasks,
    lineage_model_key: str = Path(
        ...,
        description="Unique identifier of any model from the lineage to be replaced (can be champion or challenger from the same dataset)",
        example="a66ccf8aa3634dfb5afe46cb19f79d01"
    ),
    file: UploadFile = File(
        ...,
        description="New training dataset in CSV format with features and target column"
    ),
    n: int = Form(
        3,
        ge=1,
        le=10,
        description="Number of new challenger models to train alongside the champion. Valid range: 1 to 10",
        example=3
    ),
    metric: str = Form(
        "Accuracy",
        description="Optimization metric for AutoML training. Valid options:\n* **Accuracy**: Overall correctness\n* **Precision**: Positive prediction accuracy\n* **Recall**: True positive coverage\n* **F1**: Harmonic mean of precision and recall\n* **AUC**: Area under ROC curve",
        example="Accuracy"
    ),
    db: Session = Depends(get_db)
):
    """
    ### Description
    Replaces an entire model lineage (champion and all challengers) with a new generation trained on updated data.
    
    This endpoint triggers a complete model lifecycle replacement, archiving the old lineage and creating
    a new champion with fresh challengers using the provided dataset.
    
    ### Execution Flow
    1. Receive model key from old lineage to identify dataset linkage
    2. Receive new CSV training dataset
    3. Background processing executes:
        * Find all models from the old lineage (same dataset origin)
        * Archive all old models (champion and challengers)
        * Save and validate new CSV file
        * Execute complete AutoML pipeline with new data
        * Create new champion and challenger models
        * Register lineage metadata for traceability
    """
    # Basic validations (similar to /train endpoint)
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=415, detail="Formato de arquivo inválido. Apenas .csv é aceito.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Arquivo enviado está vazio.")

    # TODO: Implementar função execute_retraining_pipeline para processamento em background
    # A lógica pesada é agendada para rodar em background
    # background_tasks.add_task(
    #     execute_retraining_pipeline,
    #     db=db,
    #     lineage_model_key=lineage_model_key,
    #     new_file_content=content,
    #     new_filename=file.filename,
    #     n_models=n,
    #     metric=metric
    # )

    logger.warning("Lineage retraining not implemented yet - feature under development")
    
    return {
        "message": "Feature under development - lineage retraining not yet implemented.",
        "lineage_model_key": lineage_model_key,
        "new_dataset": file.filename,
        "status": "pending_implementation"
    }