"""
Repository for model operations
"""
import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
from src.database.models.File import File
from src.database.models.Result import Result, ModelStage

logger = logging.getLogger(__name__)

class ModelRepository:
    """Repository for model operations"""
    
    @staticmethod
    def create(db: Session, model_data: dict) -> Result:
        """Create a new model"""
        db_result = Result(**model_data)
        db.add(db_result)
        db.commit()
        db.refresh(db_result)
        return db_result
    
    @staticmethod
    def get_by_key(db: Session, model_key: str) -> Optional[Result]:
        """Get model by key"""
        return db.query(Result).filter(Result.key == model_key).first()
    
    @staticmethod
    def get_champion_model(db: Session, file_id: Optional[int] = None) -> Optional[Result]:
        """Get champion model"""
        query = db.query(Result).filter(
            Result.stage == ModelStage.champion,
            Result.is_active == True
        )
        
        if file_id:
            query = query.filter(Result.file_id == file_id)
        
        return query.order_by(desc(Result.accuracy)).first()
    
    @staticmethod
    def get_all_models(db: Session) -> List[Result]:
        """Get all models"""
        return db.query(Result).order_by(desc(Result.created_at)).all()
    
    @staticmethod
    def get_active_models(db: Session) -> List[Result]:
        """Get active models"""
        return db.query(Result).filter(Result.is_active == True).all()
    
    @staticmethod
    def promote_to_champion(db: Session, model_key: str) -> bool:
        """
        Promote model to champion and demote current champion to challenger.
        
        Logic:
        1. Find the model to promote (must be challenger stage)
        2. Find current champion in the same lineage (file_id)
        3. Demote current champion to challenger stage
        4. Promote new model to champion stage
        
        Args:
            db: Database session
            model_key: Key of the challenger model to promote
            
        Returns:
            bool: True if promotion successful, False otherwise
        """
        try:
            new_champion = db.query(Result).filter(Result.key == model_key).first()
            if not new_champion:
                return False
            
            # Buscar o champion atual da mesma linhagem
            current_champion = db.query(Result).filter(
                Result.stage == ModelStage.champion,
                Result.file_id == new_champion.file_id,
                Result.is_active == True
            ).first()
            
            # Demote old champion to challenger (DO NOT archive)
            if current_champion:
                current_champion.stage = ModelStage.challenger
                # Keep is_active = True to allow future re-promotion
            
            # Promote new champion
            new_champion.stage = ModelStage.champion
            
            db.commit()
            return True
            
        except Exception:
            db.rollback()
            return False
    
    @staticmethod
    def archive_model(db: Session, model_key: str) -> bool:
        """Archive a model"""
        try:
            model = db.query(Result).filter(Result.key == model_key).first()
            if model:
                model.stage = ModelStage.archived
                model.is_active = False
                db.commit()
                return True
            return False
        except Exception:
            db.rollback()
            return False
    
    @staticmethod
    def get_challengers(db: Session, file_id: int, order_by_metric: str = "f1_score") -> List[Result]:
        """
        Get all challengers for a file, ordered by specified metric (best first).
        
        Args:
            db: Database session
            file_id: File ID (lineage) to filter challengers
            order_by_metric: Metric to use for ordering. Valid options:
                - "accuracy": Overall correctness
                - "f1_score": Balance precision/recall (default)
                - "precision": Minimize false positives
                - "recall": Minimize false negatives
                - "roc_auc": ROC curve area
            
        Returns:
            List of Result models ordered by specified metric descending (best first)
        """
        # Map metric name to database column
        metric_column_map = {
            "Accuracy": Result.accuracy,
            "F1": Result.f1_score,
            "Precision": Result.precision,
            "Recall": Result.recall,
            "AUC": Result.roc_auc,
            # Fallback for compatibility
            "accuracy": Result.accuracy,
            "f1_score": Result.f1_score,
            "precision": Result.precision,
            "recall": Result.recall,
            "roc_auc": Result.roc_auc
        }
        
        # Get sorting column (default: f1_score)
        order_column = metric_column_map.get(order_by_metric, Result.f1_score)
        
        return db.query(Result).filter(
            Result.file_id == file_id,
            Result.stage == ModelStage.challenger,
            Result.is_active == True
        ).order_by(desc(order_column)).all()
    
    @staticmethod
    def count_active_models_by_stage(db: Session, file_id: int) -> Dict[str, int]:
        """Count models by stage"""
        from sqlalchemy import func
        
        counts = db.query(
            Result.stage,
            func.count(Result.id).label('count')
        ).filter(
            Result.file_id == file_id,
            Result.is_active == True
        ).group_by(Result.stage).all()
        
        return {stage.value: count for stage, count in counts}
    
    @staticmethod
    def get_model_performance_summary(db: Session, file_id: int) -> Dict[str, Any]:
        """Get model performance summary"""
        models = db.query(Result).filter(
            Result.file_id == file_id,
            Result.is_active == True
        ).all()
        
        if not models:
            return {}
        
        return {
            'total_models': len(models),
            'champion_count': len([m for m in models if m.stage == ModelStage.champion]),
            'challenger_count': len([m for m in models if m.stage == ModelStage.challenger]),
            'best_accuracy': max(m.accuracy for m in models),
            'worst_accuracy': min(m.accuracy for m in models),
            'avg_accuracy': sum(m.accuracy for m in models) / len(models)
        }

    @staticmethod
    def get_by_file_id(db: Session, file_id: int) -> List[Result]:
        """Get models by file_id"""
        return db.query(Result).filter(Result.file_id == file_id).all()
    
    @staticmethod
    def get_models_by_alert(db: Session, model_key: str) -> List[Result]:
        """
        Get all models associated with an alert.
        Returns the model itself and all models from the same file_id (lineage).
        """
        base_model = db.query(Result).filter(Result.key == model_key).first()
        if not base_model:
            return []
        
        # Retorna todos os modelos ativos da mesma linhagem (file_id)
        return db.query(Result).filter(
            Result.file_id == base_model.file_id,
            Result.is_active == True
        ).all()
    
    @staticmethod
    def archive_models_from_lineage(db: Session, file_id: int) -> int:
        """
        Archive all active models from a specific file lineage.
        Returns the number of archived models.
        """
        try:
            models = db.query(Result).filter(
                Result.file_id == file_id,
                Result.is_active == True
            ).all()
            
            archived_count = 0
            for model in models:
                model.stage = ModelStage.archived
                model.is_active = False
                archived_count += 1
            
            db.commit()
            return archived_count
            
        except Exception:
            db.rollback()
            return 0
    
    @staticmethod
    def replace_lineage_with_new_champion(db: Session, old_file_id: int, new_champion_key: str) -> bool:
        """
        Replace entire model lineage with a new champion.
        Archives all old models and ensures new models keep their assigned stages.
        
        IMPORTANT: This method does NOT change stages of new models!
        New models already have stages assigned by _assign_model_stages():
        - Champion (1 model)
        - Challengers (n-1 models)
        - Archived (extras beyond n)
        
        Args:
            old_file_id: File ID of old model lineage to archive
            new_champion_key: Key of new model to promote to champion (for verification)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # 1. Verificar que o novo champion existe
            new_champion = db.query(Result).filter(Result.key == new_champion_key).first()
            if not new_champion:
                logger.error(f"New champion {new_champion_key[:8]} not found")
                db.rollback()
                return False
            
            new_file_id = new_champion.file_id
            
            # 2. IMPORTANT: DO NOT archive if same lineage!
            # (prevents archiving models that were just created)
            if old_file_id == new_file_id:
                logger.warning(f"Old and new lineage are the same (file_id={old_file_id}). Skipping archive.")
                return True
            
            # 3. Archive all old models (only from old lineage)
            archived_count = ModelRepository.archive_models_from_lineage(db, old_file_id)
            logger.info(f"Archived {archived_count} old models from lineage {old_file_id}")
            
            # 4. Validate that new models are correct (DO NOT modify, only validate)
            new_models_count = db.query(Result).filter(
                Result.file_id == new_file_id,
                Result.is_active == True
            ).count()
            
            champion_count = db.query(Result).filter(
                Result.file_id == new_file_id,
                Result.stage == ModelStage.champion,
                Result.is_active == True
            ).count()
            
            challenger_count = db.query(Result).filter(
                Result.file_id == new_file_id,
                Result.stage == ModelStage.challenger,
                Result.is_active == True
            ).count()
            
            logger.info(f"New lineage {new_file_id}: {new_models_count} active models "
                       f"({champion_count} champion + {challenger_count} challengers)")
            
            db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error replacing lineage: {e}", exc_info=True)
            db.rollback()
            return False

class FileRepository:
    """Repository for file operations"""
    
    @staticmethod
    def create(db: Session, file_data: dict) -> File:
        """Create a new file"""
        db_file = File(**file_data)
        db.add(db_file)
        db.commit()
        db.refresh(db_file)
        return db_file
    
    @staticmethod
    def get_by_id(db: Session, file_id: int) -> Optional[File]:
        """Get file by ID"""
        return db.query(File).filter(File.id == file_id).first()
    
    @staticmethod
    def get_all(db: Session) -> List[File]:
        """Get all files"""
        return db.query(File).all()