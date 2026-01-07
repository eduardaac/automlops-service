import enum
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Enum
from sqlalchemy.orm import relationship
from src.database.config import Base
from datetime import datetime

class ModelStage(enum.Enum):
    champion = "champion"
    challenger = "challenger"
    archived = "archived"

class Result(Base):
    __tablename__ = "results"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    model = Column(String, index=True)
    
    # Classification metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float) 
    roc_auc = Column(Float)   
    cross_val_roc_auc = Column(Float)
    
    # Regression metrics
    mae = Column(Float)
    rmse = Column(Float)
    r2 = Column(Float)
    mse = Column(Float)
      
    file_id = Column(Integer, ForeignKey("files.id"), index=True)
    
    pickle_path = Column(String)
    
    stage = Column(Enum(ModelStage), index=True)
    is_active = Column(Boolean, default=True, index=True)
    
    evaluation_strategy = Column(String, default="metric_drop")
    evaluation_interval = Column(Integer, default=3600)
    threshold = Column(Float, default=0.05)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    last_evaluated = Column(DateTime, default=datetime.utcnow)
    
    file = relationship("File", back_populates="results")