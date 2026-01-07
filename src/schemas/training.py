"""
Schemas for training operations
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ConfiguracaoTreinamento(BaseModel):
    """Configuration applied to training."""
    nome_arquivo: str = Field(..., description="Dataset file name")
    coluna_alvo: str = Field(..., description="Target column name")
    qtd_modelos: int = Field(..., description="Number of trained models")
    metrica_otimizacao: str = Field(..., description="Metric used for optimization")
    estrategia_monitoramento: str = Field(..., description="Monitoring strategy")
    intervalo_avaliacao: int = Field(..., description="Evaluation interval in seconds")
    threshold_degradacao: float = Field(..., description="Degradation threshold")

class RespostaTreinamento(BaseModel):
    """Training initiation response."""
    sucesso: bool = Field(True, description="Indicates operation success")
    mensagem: str = Field(..., description="Descriptive result message")
    id_job: int = Field(..., description="Unique job ID for tracking")
    configuracao: ConfiguracaoTreinamento = Field(..., description="Applied configuration")
    dataset_info: Dict[str, Any] = Field(..., description="Dataset information")
    tempo_estimado_minutos: int = Field(..., description="Estimated completion time")
    criado_em: str = Field(..., description="Creation timestamp ISO 8601")

class ErroTreinamento(BaseModel):
    """Error structure for training."""
    erro: str = Field(..., description="Error description")
    detalhes: Optional[Any] = Field(None, description="Specific error details")
    codigo: str = Field(..., description="Unique error code")
    timestamp: Optional[str] = Field(None, description="Error timestamp")

class StatusTreinamento(BaseModel):
    """Training job status."""
    job_id: int = Field(..., description="Job ID")
    status: str = Field(..., description="Current status: PENDING, RUNNING, COMPLETED, FAILED")
    detalhes: str = Field(..., description="Current status details")
    modelos_criados: List[Dict[str, Any]] = Field(..., description="List of created models")
    tempo_execucao_segundos: int = Field(..., description="Total execution time")
    criado_em: str = Field(..., description="Job creation date")
    atualizado_em: str = Field(..., description="Last status update")

class InfoModelo(BaseModel):
    """Model information supporting both classification and regression."""
    id: int = Field(..., description="Unique database model ID")
    chave: str = Field(..., description="SHA256 model key for predictions")
    nome_modelo: str = Field(..., description="Algorithm name (e.g., RandomForestClassifier, XGBRegressor)")
    
    # Classification metrics (null for regression models)
    acuracia: Optional[float] = Field(
        None, 
        description="Accuracy (0.0-1.0) - Classification only"
    )
    precisao: Optional[float] = Field(
        None, 
        description="Precision (0.0-1.0) - Classification only"
    )
    recall: Optional[float] = Field(
        None, 
        description="Recall (0.0-1.0) - Classification only"
    )
    f1_score: Optional[float] = Field(
        None, 
        description="F1-Score (0.0-1.0) - Classification only"
    )
    roc_auc: Optional[float] = Field(
        None, 
        description="ROC AUC (0.0-1.0) - Classification only"
    )
    
    # Regression metrics (null for classification models)
    mae: Optional[float] = Field(
        None, 
        description="Mean Absolute Error - Regression only (lower is better)"
    )
    rmse: Optional[float] = Field(
        None, 
        description="Root Mean Squared Error - Regression only (lower is better)"
    )
    r2: Optional[float] = Field(
        None, 
        description="R² Score (0.0-1.0) - Regression only (higher is better)"
    )
    mse: Optional[float] = Field(
        None, 
        description="Mean Squared Error - Regression only (lower is better)"
    )
    
    estagio: str = Field(
        ..., 
        description="Deployment stage: champion (production), challenger (testing), archived (inactive)"
    )
    ativo: bool = Field(..., description="Whether model is active and available for predictions")
    criado_em: str = Field(..., description="Creation timestamp (ISO 8601)")
    arquivo_dataset: str = Field(..., description="Original training dataset filename")

class ListaModelos(BaseModel):
    """System models list."""
    total: int = Field(..., description="Total models")
    modelos: List[InfoModelo] = Field(..., description="Models list")

class RespostaPromocao(BaseModel):
    """Model promotion response."""
    sucesso: bool = Field(True, description="Indicates operation success")
    mensagem: str = Field(..., description="Status message")
    modelo_promovido: str = Field(..., description="Promoted model key")
    modelo_anterior: Optional[str] = Field(None, description="Previous model key")
    timestamp: str = Field(..., description="Operation timestamp")

class RespostaRetreinamento(BaseModel):
    """Retraining from alert response."""
    message: str = Field(..., description="Status message")
    new_job_id: int = Field(..., description="New retraining job ID")
    alert_resolved_id: int = Field(..., description="Resolved alert ID")
    old_model_id: str = Field(..., description="Old model key")
    old_model_name: str = Field(..., description="Old model name")
    timestamp: str = Field(..., description="Operation timestamp")