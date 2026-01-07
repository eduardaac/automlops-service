from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from src.database.config import Base
from datetime import datetime

class File(Base):
    __tablename__ = "files"
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    content_path = Column(String, nullable=False)
    size = Column(Integer, nullable=True)
    content_type = Column(String, nullable=True)
    target_column = Column(String, nullable=True)
    optimization_metric = Column(String, nullable=True, default="Accuracy")  # Metric used in training
    created_at = Column(DateTime, default=datetime.utcnow)
    
    results = relationship("Result", back_populates="file")