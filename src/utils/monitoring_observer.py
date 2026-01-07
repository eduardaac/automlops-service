"""
Model Monitoring Observer - Decision Support System
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from src.database.config import SessionLocal
from src.database.models.Alert import Alert, AlertType, AlertStatus
from src.database.models.Result import Result as DBResult, ModelStage
from src.utils.observer_pattern import Observer, Subject

logger = logging.getLogger(__name__)

class MonitoringObserver(Observer):
    """
    Observer that detects risk events and creates alerts for human decision support.
    System converted from automatic actions to problem reporter.
    """
    
    def __init__(self):
        self.alerts_created = 0
        logger.info("MonitoringObserver initialized")
    
    async def update(self, event_data: Dict[str, Any], subject: Subject) -> None:
        """
        Called when a health event is detected.
        Creates alerts based on health index degradation or system errors.

        Args:
            event_data: Event data containing event_type and metrics.
            subject: Subject that triggered the event.
        """
        try:
            event_type = event_data.get('event_type')
            model_key = event_data.get('model_key', 'unknown')
            
            # TRIGGER 1: Model degradation (health_index drops below threshold)
            if event_type == "model_degradation_detected":
                health_index = event_data.get('health_index', 0.0)
                threshold = event_data.get('threshold', 0.7)
                prediction_count = event_data.get('prediction_count', 0)
                elevated_risk_count = event_data.get('elevated_risk_count', 0)
                risk_components = event_data.get('risk_components', {})
                
                # Determine severity based on health index
                if health_index < 0.5:
                    severity = "CRITICAL"
                    action = "URGENT RETRAINING REQUIRED"
                elif health_index < 0.6:
                    severity = "HIGH"
                    action = "Retraining strongly recommended"
                else:
                    severity = "MODERATE"
                    action = "Intensified monitoring and root cause analysis"
                
                details = (
                    f"Performance Degradation Detected - {severity}\n\n"
                    f"Health Metrics:\n"
                    f"  • Health Index: {health_index:.3f} (Threshold: {threshold:.2f})\n"
                    f"  • Observations: {prediction_count} predictions\n"
                    f"  • Confirmed Signals: {elevated_risk_count}/3 elevated components\n\n"
                    f"Risk Components (threshold: 0.30):\n"
                    f"  • Confidence Risk: {risk_components.get('confidence_risk', 0.0):.3f} "
                    f"{'ELEVATED' if risk_components.get('confidence_risk', 0.0) > 0.3 else 'OK'}\n"
                    f"  • Drift Risk: {risk_components.get('drift_risk', 0.0):.3f} "
                    f"{'ELEVATED' if risk_components.get('drift_risk', 0.0) > 0.3 else 'OK'}\n"
                    f"  • Anomaly Risk: {risk_components.get('anomaly_risk', 0.0):.3f} "
                    f"{'ELEVATED' if risk_components.get('anomaly_risk', 0.0) > 0.3 else 'OK'}\n\n"
                    f"Recommended Action: {action}\n"
                    f"Timestamp: {datetime.utcnow().isoformat()}"
                )
                
                await self._create_risk_alert(model_key, details, AlertType.performance_degradation)
                self.alerts_created += 1
                
                # Check if there's a better challenger available
                await self._check_and_alert_better_challenger(model_key, health_index)
            
            # TRIGGER 2: System volume overload (volume_risk) - DISABLED
            # Volume alerts removed per user request
            # elif event_type == "system_error":
            #     volume_risk = event_data.get('volume_risk', 0.0)
            #     count = event_data.get('prediction_count', 0)
            #     
            #     details = (
            #         f"⚠️ Alerta de volume do sistema.\n"
            #         f"📊 Volume Risk atingiu {volume_risk:.3f} ({count} predições)\n\n"
            #         f"Ação Recomendada: Verificar capacidade do sistema e considerar scaling.\n"
            #         f"Timestamp: {datetime.utcnow().isoformat()}"
            #     )
            #     
            #     await self._create_risk_alert(model_key, details, AlertType.system_error)
            #     self.alerts_created += 1
                
        except Exception as e:
            logger.error(f"MonitoringObserver error: {e}", exc_info=True)
    
    async def _create_risk_alert(self, model_key: str, details: str, alert_type: AlertType):
        """
        Creates an alert in the database with deduplication by LINEAGE (not per model).
        
        Prevents duplicate alerts for models from the same training experiment (same file_id).
        Only one alert per lineage/experiment is created.

        Args:
            model_key: Model key.
            details: Formatted alert details.
            alert_type: Type of alert (performance_degradation only).
        """
        db: Session = SessionLocal()
        try:
            # 1. Get model to find its lineage (file_id)
            model = db.query(DBResult).filter(DBResult.key == model_key).first()
            if not model:
                logger.warning(f"Model {model_key[:8]} not found for alert creation")
                return
            
            # 2. Check if alert already exists for ANY model in the same lineage
            lineage_models = db.query(DBResult).filter(
                DBResult.file_id == model.file_id,
                DBResult.is_active == True
            ).all()
            
            lineage_keys = [m.key for m in lineage_models]
            
            existing_alert = db.query(Alert).filter(
                Alert.model_key.in_(lineage_keys),
                Alert.alert_type == alert_type,
                Alert.status.in_([AlertStatus.open, AlertStatus.acknowledged])
            ).first()
            
            if existing_alert:
                # Update existing alert (no duplicate by lineage)
                existing_alert.details = details
                existing_alert.updated_at = datetime.utcnow()
                logger.info(
                    f"Updated existing {alert_type.value} alert for lineage "
                    f"(file_id: {model.file_id}, {len(lineage_models)} models)"
                )
            else:
                # Create new alert associated with CHAMPION of lineage (if exists)
                # This avoids creating alerts for each individual model
                champion = next((m for m in lineage_models if m.stage == ModelStage.champion), None)
                alert_model_key = champion.key if champion else model_key
                
                new_alert = Alert(
                    model_key=alert_model_key,  # Associate with lineage champion
                    alert_type=alert_type,
                    status=AlertStatus.open,
                    details=details,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(new_alert)
                logger.info(
                    f"Created new {alert_type.value} alert for lineage "
                    f"(file_id: {model.file_id}, {len(lineage_models)} models, "
                    f"associated to: {alert_model_key[:8]})"
                )
            
            db.commit()
        except Exception as e:
            logger.error(f"Alert creation error: {e}")
            db.rollback()
        finally:
            db.close()
    
    async def _check_and_alert_better_challenger(self, champion_key: str, champion_health: float):
        """
        Checks if there's a challenger performing better than the current champion.
        Uses ROBUST CRITERIA (not just accuracy) for promotion recommendation.
        
        Criteria for challenger promotion:
        1. F1 Score AND Accuracy must be better than champion
        2. Challenger's health_index must be > 0.7 (healthy)
        3. Challenger must show consistent improvement (not marginal)
        
        Args:
            champion_key: Current champion model key
            champion_health: Health index of the champion
        """
        db: Session = SessionLocal()
        try:
            # 1. Fetch the champion model
            champion = db.query(DBResult).filter(DBResult.key == champion_key).first()
            if not champion:
                return
            
            # 2. Get challengers from the same lineage
            from src.database.models.Result import ModelStage
            from src.database.models.PerformanceLog import PerformanceLog
            
            challengers = db.query(DBResult).filter(
                DBResult.file_id == champion.file_id,
                DBResult.stage == ModelStage.challenger,
                DBResult.is_active == True
            ).order_by(DBResult.f1_score.desc()).all()  # Order by F1 (more robust than accuracy)
            
            if not challengers:
                logger.info(f"No challengers available for champion {champion_key[:8]}")
                return
            
            # 3. Apply ROBUST CRITERIA for each challenger
            for challenger in challengers:
                # Get latest health_index for challenger
                challenger_latest_perf = db.query(PerformanceLog).filter(
                    PerformanceLog.model_key == challenger.key,
                    PerformanceLog.health_index.isnot(None)
                ).order_by(PerformanceLog.timestamp.desc()).first()
                
                challenger_health = challenger_latest_perf.health_index if challenger_latest_perf else None
                
                # CRITERION 1: F1 Score must be better (primary metric for classification)
                f1_better = challenger.f1_score > champion.f1_score
                
                # CRITERION 2: Accuracy must also be better (secondary confirmation)
                accuracy_better = challenger.accuracy > champion.accuracy
                
                # CRITERION 3: Challenger must be healthy (health_index > 0.7)
                challenger_healthy = challenger_health is not None and challenger_health > 0.7
                
                # CRITERION 4: Improvement must be significant (> 1% relative improvement)
                f1_improvement = ((challenger.f1_score - champion.f1_score) / champion.f1_score) * 100
                significant_improvement = f1_improvement > 1.0
                
                # ALL CRITERIA MUST BE MET
                if f1_better and accuracy_better and challenger_healthy and significant_improvement:
                    # Calculate improvements
                    f1_diff = ((challenger.f1_score - champion.f1_score) / champion.f1_score) * 100
                    accuracy_diff = ((challenger.accuracy - champion.accuracy) / champion.accuracy) * 100
                    roc_diff = ((challenger.roc_auc - champion.roc_auc) / champion.roc_auc) * 100 if champion.roc_auc else 0
                    
                    details = (
                        f"Better-performing challenger available!\n\n"
                        f"Current Champion:\n"
                        f"  • Model: {champion.model}\n"
                        f"  • F1 Score: {champion.f1_score:.4f}\n"
                        f"  • Accuracy: {champion.accuracy:.4f}\n"
                        f"  • ROC AUC: {champion.roc_auc:.4f}\n"
                        f"  • Health Index: {champion_health:.3f}\n\n"
                        f"Recommended Challenger:\n"
                        f"  • Model: {challenger.model}\n"
                        f"  • F1 Score: {challenger.f1_score:.4f} (+{f1_diff:.2f}%)\n"
                        f"  • Accuracy: {challenger.accuracy:.4f} (+{accuracy_diff:.2f}%)\n"
                        f"  • ROC AUC: {challenger.roc_auc:.4f} (+{roc_diff:.2f}%)\n"
                        f"  • Health Index: {challenger_health:.3f}\n"
                        f"  • Key: {challenger.key[:16]}...\n\n"
                        f"Robust Criteria Met:\n"
                        f"  - F1 Score improvement: +{f1_diff:.2f}%\n"
                        f"  - Accuracy improvement: +{accuracy_diff:.2f}%\n"
                        f"  - Challenger health: {challenger_health:.3f} (> 0.7)\n"
                        f"  - Significant improvement: {f1_improvement:.2f}% (> 1%)\n\n"
                        f"Recommended Action:\n"
                        f"  Promote challenger to production using:\n"
                        f"  POST /human-actions/alerts/{{alert_id}}/promote-challenger\n\n"
                        f"Timestamp: {datetime.utcnow().isoformat()}"
                    )
                    
                    # 4. Create challenger available alert (only for champion's lineage, not per model)
                    await self._create_challenger_alert(
                        model_key=champion_key,  # Alert associado ao champion (não duplica por challenger)
                        details=details,
                        challenger_key=challenger.key
                    )
                    
                    logger.info(
                        f"Challenger alert created: {challenger.model} "
                        f"(F1: {challenger.f1_score:.4f}, Health: {challenger_health:.3f}) > "
                        f"{champion.model} (F1: {champion.f1_score:.4f}, Health: {champion_health:.3f})"
                    )
                    
                    # Only create ONE alert per lineage (stop after first valid challenger)
                    break
                else:
                    logger.debug(
                        f"Challenger {challenger.model} did not meet robust criteria: "
                        f"F1={f1_better}, Acc={accuracy_better}, Health={challenger_healthy}, "
                        f"Significant={significant_improvement}"
                    )
                
        except Exception as e:
            logger.error(f"Error checking for better challenger: {e}", exc_info=True)
        finally:
            db.close()
    
    async def _create_challenger_alert(self, model_key: str, details: str, challenger_key: str):
        """
        Creates a specific alert for an available challenger.
        
        Args:
            model_key: Champion model key
            details: Formatted alert details
            challenger_key: Recommended challenger model key
        """
        db: Session = SessionLocal()
        try:
            # Check if there's already an open challenger alert for this model
            existing_alert = db.query(Alert).filter(
                Alert.model_key == model_key,
                Alert.alert_type == AlertType.challenger_available,
                Alert.status.in_([AlertStatus.open, AlertStatus.acknowledged])
            ).first()
            
            if existing_alert:
                # Update existing alert
                existing_alert.details = details
                existing_alert.updated_at = datetime.utcnow()
                logger.info(f"Updated existing challenger alert for model {model_key[:8]}")
            else:
                # Create new alert
                new_alert = Alert(
                    model_key=model_key,
                    alert_type=AlertType.challenger_available,
                    status=AlertStatus.open,
                    details=details,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(new_alert)
                self.alerts_created += 1
                logger.info(f"Created new challenger alert for model {model_key[:8]}")
            
            db.commit()
        except Exception as e:
            logger.error(f"Challenger alert creation error: {e}", exc_info=True)
            db.rollback()
        finally:
            db.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Returns observer statistics.

        Returns:
            Dict with statistics data.
        """
        return {
            "alerts_created": self.alerts_created,
            "observer_type": "decision_support",
            "mode": "reporter",
            "timestamp": datetime.utcnow().isoformat()
        }

observador_monitoramento = MonitoringObserver()