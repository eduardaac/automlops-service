"""
Model Monitoring Service - Long-Term Trend Analysis
"""
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from src.database.config import SessionLocal
from src.database.models.Result import Result as DBResult
from src.database.models.Alert import Alert, AlertType, AlertStatus
from src.services.performance_service import performance_service

logger = logging.getLogger(__name__)

async def monitor_long_term_trends():
    """
    Execute long-term monitoring routine.
    This function should be called by a scheduler (e.g., every 24 hours).
    """
    logger.info("Starting long-term trend monitoring routine...")
    db: Session = SessionLocal()
    try:
        active_models = db.query(DBResult).filter(DBResult.is_active == True).all()
        
        if not active_models:
            logger.info("No active models found for long-term monitoring.")
            return

        logger.info(f"Analyzing trends for {len(active_models)} active models...")
        
        for model in active_models:
            await _check_long_term_degradation(db, model)
            
    except Exception as e:
        logger.error(f"Critical error during trend monitoring routine: {e}", exc_info=True)
    finally:
        db.close()
        logger.info("Long-term trend monitoring routine completed.")

async def _check_long_term_degradation(db: Session, model: DBResult):
    """
    Use performance_service to analyze model trend and create alert if necessary.
    """
    try:
        trends = performance_service.get_performance_trends(db, model.key, days=30)
        
        trend_direction = trends.get("trend")
        
        if trend_direction == "degrading":
            magnitude = trends.get("magnitude", 0)
            recent_avg = trends.get("recent_average", 0)
            historical_avg = trends.get("historical_average", 0)
            
            logger.warning(
                f"Degradation trend detected for model {model.key[:8]}! "
                f"Magnitude: {magnitude:.4f}"
            )
            
            await _create_degradation_trend_alert(db, model, magnitude, recent_avg, historical_avg)
            
    except Exception as e:
        logger.error(f"Error checking degradation trend for model {model.key[:8]}: {e}")

async def _create_degradation_trend_alert(
    db: Session, 
    model: DBResult, 
    magnitude: float, 
    recent_avg: float, 
    historical_avg: float
):
    """
    Create or update performance degradation alert in database.
    """
    try:
        existing_alert = db.query(Alert).filter(
            Alert.model_key == model.key,
            Alert.alert_type == AlertType.performance_degradation,
            Alert.status.in_([AlertStatus.open, AlertStatus.acknowledged])
        ).first()
        
        details = (
            f"Long-term degradation trend detected. "
            f"Drop magnitude: {magnitude:.2%}. "
            f"Historical average performance: {historical_avg:.4f}, "
            f"Recent average performance: {recent_avg:.4f}."
        )
        
        if existing_alert:
            existing_alert.details = details
            existing_alert.updated_at = datetime.utcnow()
            logger.info(f"Degradation alert updated for model {model.key[:8]}.")
        else:
            new_alert = Alert(
                model_key=model.key,
                alert_type=AlertType.performance_degradation,
                status=AlertStatus.open,
                details=details,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(new_alert)
            logger.warning(f"New long-term degradation alert created for model {model.key[:8]}.")
        
        db.commit()
        
    except Exception as e:
        logger.error(f"Error creating or updating degradation alert for model {model.key}: {e}")
        db.rollback()