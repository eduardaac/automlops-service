"""
Alert Management Service
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc

from src.database.models.Alert import Alert, AlertType, AlertStatus
from src.database.models.Result import Result as DBResult

logger = logging.getLogger(__name__)

class AlertService:
    """Service for system alert management"""
    
    def get_all_alerts(self, db: Session, limit: int = 100, offset: int = 0) -> List[Alert]:
        """Get all alerts, sorted by creation date."""
        return db.query(Alert).order_by(desc(Alert.created_at)).offset(offset).limit(limit).all()
    
    def get_open_alerts(self, db: Session) -> List[Alert]:
        """Get only open alerts."""
        return db.query(Alert).filter(Alert.status == AlertStatus.open).order_by(desc(Alert.created_at)).all()
    
    def get_alerts_by_model(self, db: Session, model_key: str) -> List[Alert]:
        """Get alerts for a specific model."""
        return db.query(Alert).filter(Alert.model_key == model_key).order_by(desc(Alert.created_at)).all()
    
    def get_alerts_by_type(self, db: Session, alert_type: AlertType) -> List[Alert]:
        """Get alerts by type."""
        return db.query(Alert).filter(Alert.alert_type == alert_type).order_by(desc(Alert.created_at)).all()
    
    def acknowledge_alert(self, db: Session, alert_id: int) -> bool:
        """Mark alert as acknowledged."""
        try:
            alert = db.query(Alert).filter(Alert.id == alert_id).first()
            if not alert:
                return False
            
            alert.status = AlertStatus.acknowledged
            alert.updated_at = datetime.utcnow()
            
            db.commit()
            logger.info(f"Alert {alert_id} marked as acknowledged")
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            db.rollback()
            return False
    
    def resolve_alert(self, db: Session, alert_id: int, resolution_notes: str = None) -> bool:
        """Mark alert as resolved."""
        try:
            alert = db.query(Alert).filter(Alert.id == alert_id).first()
            if not alert:
                return False
            
            alert.status = AlertStatus.resolved
            alert.resolved_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            
            if resolution_notes:
                alert.details += f"\n\nResolution: {resolution_notes}"
            
            db.commit()
            logger.info(f"Alert {alert_id} marked as resolved")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            db.rollback()
            return False
    
    def get_alert_summary(self, db: Session) -> Dict[str, Any]:
        """Get system alert summary."""
        try:
            total_alerts = db.query(Alert).count()
            open_alerts = db.query(Alert).filter(Alert.status == AlertStatus.open).count()
            acknowledged_alerts = db.query(Alert).filter(Alert.status == AlertStatus.acknowledged).count()
            resolved_alerts = db.query(Alert).filter(Alert.status == AlertStatus.resolved).count()
            
            alerts_by_type = {}
            for alert_type in AlertType:
                count = db.query(Alert).filter(Alert.alert_type == alert_type).count()
                alerts_by_type[alert_type.value] = count
            
            yesterday = datetime.utcnow() - timedelta(days=1)
            recent_alerts = db.query(Alert).filter(Alert.created_at >= yesterday).count()
            
            return {
                "total_alerts": total_alerts,
                "open_alerts": open_alerts,
                "acknowledged_alerts": acknowledged_alerts,
                "resolved_alerts": resolved_alerts,
                "alerts_by_type": alerts_by_type,
                "recent_alerts_24h": recent_alerts,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting alert summary: {e}", exc_info=True)
            # Return default structure to avoid breaking the API
            return {
                "total_alerts": 0,
                "open_alerts": 0,
                "acknowledged_alerts": 0,
                "resolved_alerts": 0,
                "alerts_by_type": {},
                "recent_alerts_24h": 0,
                "timestamp": datetime.utcnow().isoformat()
            }

alert_service = AlertService()