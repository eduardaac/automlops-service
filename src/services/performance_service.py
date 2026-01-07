"""
Performance Monitoring Service
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from src.database.models.PerformanceLog import PerformanceLog
from src.database.models.Result import Result as DBResult

logger = logging.getLogger(__name__)

class PerformanceService:
    """Service for model performance analysis"""
    
    def get_model_performance_history(self, db: Session, model_key: str, days: int = 30) -> List[PerformanceLog]:
        """Get model performance history."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        return db.query(PerformanceLog).filter(
            PerformanceLog.model_key == model_key,
            PerformanceLog.timestamp >= start_date
        ).order_by(PerformanceLog.timestamp).all()
    
    def get_model_latest_performance(self, db: Session, model_key: str) -> Optional[PerformanceLog]:
        """Get most recent model performance."""
        return db.query(PerformanceLog).filter(
            PerformanceLog.model_key == model_key
        ).order_by(desc(PerformanceLog.timestamp)).first()
    
    def get_performance_comparison(self, db: Session, model_keys: List[str], days: int = 7) -> Dict[str, Any]:
        """Compare performance between models."""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        comparison = {}
        
        for model_key in model_keys:
            latest_perf = db.query(PerformanceLog).filter(
                PerformanceLog.model_key == model_key,
                PerformanceLog.timestamp >= start_date
            ).order_by(desc(PerformanceLog.timestamp)).first()
            
            if latest_perf:
                comparison[model_key] = {
                    "accuracy": latest_perf.accuracy,
                    "precision": latest_perf.precision,
                    "recall": latest_perf.recall,
                    "f1_score": latest_perf.f1_score,
                    "last_evaluation": latest_perf.timestamp.isoformat()
                }
        
        return comparison
    
    def get_performance_trends(self, db: Session, model_key: str, days: int = 30) -> Dict[str, Any]:
        """Calculate performance trends."""
        performance_history = self.get_model_performance_history(db, model_key, days)
        
        if len(performance_history) < 2:
            return {"trend": "insufficient_data", "message": "Insufficient data to calculate trend"}
        
        recent_accuracy = [p.accuracy for p in performance_history[-5:] if p.accuracy is not None]
        older_accuracy = [p.accuracy for p in performance_history[:5] if p.accuracy is not None]
        
        if not recent_accuracy or not older_accuracy:
            return {"trend": "insufficient_data", "message": "Insufficient data to calculate trend"}
        
        recent_avg = sum(recent_accuracy) / len(recent_accuracy)
        older_avg = sum(older_accuracy) / len(older_accuracy)
        
        trend_direction = "improving" if recent_avg > older_avg else "degrading" if recent_avg < older_avg else "stable"
        trend_magnitude = abs(recent_avg - older_avg)
        
        return {
            "trend": trend_direction,
            "magnitude": trend_magnitude,
            "recent_average": recent_avg,
            "historical_average": older_avg,
            "total_evaluations": len(performance_history),
            "evaluation_period_days": days
        }
    
    def get_system_performance_summary(self, db: Session) -> Dict[str, Any]:
        """Get system performance summary."""
        try:
            yesterday = datetime.utcnow() - timedelta(days=1)
            recent_logs = db.query(PerformanceLog).filter(
                PerformanceLog.timestamp >= yesterday
            ).count()
            
            models_with_perf = db.query(func.count(func.distinct(PerformanceLog.model_key))).scalar()
            
            avg_accuracy = db.query(func.avg(PerformanceLog.accuracy)).filter(
                PerformanceLog.timestamp >= yesterday,
                PerformanceLog.accuracy.isnot(None)
            ).scalar() or 0
            
            return {
                "recent_evaluations_24h": recent_logs,
                "models_evaluated": models_with_perf,
                "system_average_accuracy": float(avg_accuracy),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}

performance_service = PerformanceService()