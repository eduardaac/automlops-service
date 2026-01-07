"""
Schemas for monitoring operations
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class RespostaDrift(BaseModel):
    """Drift detection response"""
    drift_detectado: bool = Field(..., description="If drift was detected")
    score_drift: float = Field(..., description="Drift score (0-1)")
    dataset_original: str = Field(..., description="Original dataset name")
    qtd_instancias_comparadas: int = Field(..., description="Number of compared instances")
    colunas_com_drift: List[str] = Field(..., description="Columns with drift")
    detalhes_drift: Dict[str, float] = Field(default_factory=dict, description="Drift score per column")
    threshold_usado: float = Field(..., description="Threshold used for detection")
    timestamp: str = Field(..., description="Analysis timestamp")

class MetricasMonitoramento(BaseModel):
    """System monitoring metrics"""
    total_modelos: int = Field(..., description="Total models in system")
    modelos_ativos: int = Field(..., description="Active models")
    modelos_champion: int = Field(..., description="Champion models")
    modelos_challenger: int = Field(..., description="Challenger models")
    predicoes_realizadas: int = Field(..., description="Total predictions made")
    predicoes_ultimo_periodo: int = Field(..., description="Predictions in last hour")
    tempo_medio_resposta_ms: float = Field(..., description="Average response time")
    latencia_p95_ms: float = Field(..., description="P95 latency in ms")
    taxa_erro_pct: float = Field(..., description="Error rate in %")
    uptime_segundos: float = Field(..., description="Uptime")
    eventos_monitoramento: int = Field(..., description="Monitoring events triggered")
    drift_detectado: bool = Field(..., description="If drift is detected")
    cpu_percent: Optional[float] = Field(None, description="CPU usage %")
    memoria_percent: Optional[float] = Field(None, description="Memory usage %")

class StatusSistema(BaseModel):
    """System overall status"""
    status: str = Field(..., description="System status")
    versao: str = Field(..., description="API version")
    banco_dados: str = Field(..., description="Database status")
    kafka: str = Field(..., description="Kafka status")
    prometheus: str = Field(..., description="Prometheus status")
    workers_ativos: int = Field(..., description="Active workers")
    timestamp: str = Field(..., description="Status timestamp")
    detalhes: Dict[str, Any] = Field(default_factory=dict, description="Additional details")

class HealthCheck(BaseModel):
    """System health check"""
    healthy: bool = Field(..., description="If system is healthy")
    services: Dict[str, str] = Field(..., description="Services status")
    timestamp: str = Field(..., description="Check timestamp")
    uptime: float = Field(..., description="Uptime in seconds")

class ModelMetrics(BaseModel):
    """Model-specific metrics"""
    model_key: str = Field(..., description="Model key")
    model_name: str = Field(..., description="Model name")
    stage: str = Field(..., description="Model stage")
    accuracy: float = Field(..., description="Current accuracy")
    precision: Optional[float] = Field(None, description="Precision")
    recall: Optional[float] = Field(None, description="Recall")
    f1_score: Optional[float] = Field(None, description="F1-Score")
    predictions_count: int = Field(..., description="Total predictions")
    avg_latency_ms: float = Field(..., description="Average latency")
    last_prediction: Optional[str] = Field(None, description="Last prediction")
    created_at: str = Field(..., description="Creation date")

class AlertaMonitoramento(BaseModel):
    """Monitoring alert"""
    tipo: str = Field(..., description="Alert type")
    severidade: str = Field(..., description="Severity (low, medium, high, critical)")
    mensagem: str = Field(..., description="Alert message")
    modelo_afetado: Optional[str] = Field(None, description="Affected model")
    metrica: str = Field(..., description="Metric that triggered alert")
    valor_atual: float = Field(..., description="Current metric value")
    threshold: float = Field(..., description="Configured threshold")
    timestamp: str = Field(..., description="Alert timestamp")
    resolvido: bool = Field(default=False, description="If alert was resolved")

class DashboardSummary(BaseModel):
    """Dashboard summary"""
    active_models: int = Field(..., description="Active models")
    champion_models: int = Field(..., description="Champion models")
    challenger_models: int = Field(..., description="Challenger models")
    predictions_rate: float = Field(..., description="Predictions rate/sec")
    avg_latency_ms: float = Field(..., description="Average latency")
    p95_latency_ms: float = Field(..., description="P95 latency")
    error_rate: float = Field(..., description="Error rate")
    drift_alerts: int = Field(..., description="Active drift alerts")
    system_health: str = Field(..., description="System health")

class PrometheusMetrics(BaseModel):
    """Metrics in Prometheus format"""
    content: str = Field(..., description="Metrics content")
    content_type: str = Field(default="text/plain", description="Content type")

class ConfiguracaoMonitoramento(BaseModel):
    """Monitoring configuration"""
    drift_threshold: float = Field(default=0.1, description="Drift detection threshold")
    accuracy_threshold: float = Field(default=0.8, description="Minimum accuracy threshold")
    latency_threshold_ms: float = Field(default=1000, description="Maximum latency threshold")
    error_rate_threshold: float = Field(default=0.05, description="Maximum error rate threshold")
    intervalo_verificacao: int = Field(default=300, description="Verification interval in seconds")
    alertas_habilitados: bool = Field(default=True, description="If alerts are enabled")