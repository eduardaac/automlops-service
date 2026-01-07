from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from src.database.config import Base
from datetime import datetime

class PerformanceLog(Base):
    __tablename__ = "performance_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_key = Column(String, ForeignKey("results.key"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Classification metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    
    # Regression metrics
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    r2 = Column(Float, nullable=True)
    mse = Column(Float, nullable=True)
    
    # HEALTH INDEX: MÉTRICA UNIFICADA DO SISTEMA
    health_index = Column(Float, nullable=True, index=True)
    
    data_drift_score = Column(Float, nullable=True)
    concept_drift_score = Column(Float, nullable=True)
    
    evaluation_type = Column(String, default="scheduled")
    sample_size = Column(Integer, nullable=True)
    
    result = relationship("Result")