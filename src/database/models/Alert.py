import enum
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
from src.database.config import Base
from datetime import datetime

class AlertType(enum.Enum):
    performance_degradation = "Performance Degradation"  # Alert for retraining requirement
    challenger_available = "Challenger Available"        # Alert for switching requirement

class AlertStatus(enum.Enum):
    open = "Open"
    acknowledged = "Acknowledged"
    resolved = "Resolved"
    closed = "Closed"

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    model_key = Column(String, ForeignKey("results.key"), nullable=False, index=True)
    alert_type = Column(Enum(AlertType), nullable=False)
    status = Column(Enum(AlertStatus), default=AlertStatus.open)
    details = Column(Text, nullable=True)
    
    # Retraining job associated with the alert
    retraining_job_id = Column(Integer, ForeignKey("files.id"), nullable=True, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    
    result = relationship("Result")
    retraining_job = relationship("File", foreign_keys=[retraining_job_id])