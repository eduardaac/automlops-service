"""
Retraining Service - Manages model retraining from alerts
"""
import logging
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session

from src.database.models.Alert import Alert, AlertStatus
from src.database.models.Result import Result as DBResult, ModelStage
from src.database.models.File import File as DBFile
from src.repositories.model_repository import ModelRepository
from src.services.training_service import create_new_models_pipeline

logger = logging.getLogger(__name__)

class RetrainingService:
    """Service for handling model retraining triggered by alerts"""
    
    @staticmethod
    async def complete_retraining_job(db: Session, job_id: int, alert_id: int) -> bool:
        """
        Complete retraining job after training finishes.
        
        This is the critical post-training logic that:
        1. Finds the best new model from the retraining job
        2. Archives all old models from the alert lineage
        3. Promotes the new best model to champion
        4. Resolves the alert
        
        Args:
            db: Database session
            job_id: Retraining job ID (file_id)
            alert_id: Original alert ID that triggered retraining
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # 1. Buscar o alerta original
            alert = db.query(Alert).filter(Alert.id == alert_id).first()
            if not alert:
                logger.error(f"Alert {alert_id} not found")
                return False
            
            # 2. Buscar o modelo antigo associado ao alerta
            old_model = db.query(DBResult).filter(DBResult.key == alert.model_key).first()
            if not old_model:
                logger.error(f"Old model {alert.model_key} not found")
                return False
            
            old_file_id = old_model.file_id
            
            # 3. Buscar os novos modelos treinados no job de retreinamento
            model_repo = ModelRepository()
            new_models = model_repo.get_by_file_id(db, job_id)
            
            if not new_models:
                logger.error(f"No models found for retraining job {job_id}")
                return False
            
            # 4. Identificar o melhor modelo (novo champion)
            new_champion = max(new_models, key=lambda m: m.accuracy or 0)
            
            logger.info(f"Retraining completed. New champion: {new_champion.model} "
                       f"(accuracy: {new_champion.accuracy:.4f})")
            
            # 5. Substituir linhagem antiga pela nova
            success = model_repo.replace_lineage_with_new_champion(
                db=db,
                old_file_id=old_file_id,
                new_champion_key=new_champion.key
            )
            
            if not success:
                logger.error("Failed to replace model lineage")
                return False
            
            # 6. Atualizar o alerta como resolvido
            alert.status = AlertStatus.resolved
            alert.resolved_at = datetime.utcnow()
            alert.updated_at = datetime.utcnow()
            alert.details += (
                f"\n\n[AUTO-RESOLVED] Retraining completed successfully.\n"
                f"Old model lineage (file_id={old_file_id}) archived.\n"
                f"New champion: {new_champion.model} (key={new_champion.key[:8]}...)\n"
                f"New accuracy: {new_champion.accuracy:.4f}\n"
                f"Resolved at: {datetime.utcnow().isoformat()}"
            )
            
            db.commit()
            
            logger.info(f"Alert {alert_id} resolved. Old models archived, "
                       f"new champion promoted: {new_champion.key[:8]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"Error completing retraining job: {e}", exc_info=True)
            db.rollback()
            return False

retraining_service = RetrainingService()
