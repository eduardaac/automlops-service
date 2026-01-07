"""
Model Management Service
"""
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from pathlib import Path

from src.repositories.model_repository import ModelRepository, FileRepository
from src.database.models.Result import Result as DBResult, ModelStage
from src.schemas.training import InfoModelo, ListaModelos, RespostaPromocao
from src.middleware.metrics_middleware import track_model_promotion

logger = logging.getLogger(__name__)

class ModelService:
    """Service for model management"""
    
    def __init__(self):
        self.model_repo = ModelRepository()
        self.file_repo = FileRepository()
    
    def get_all_models(self, db: Session) -> ListaModelos:
        """
        Get list of all models.
        
        Args:
            db: Database session
            
        Returns:
            Formatted model list
        """
        modelos = self.model_repo.get_all_models(db)
        
        modelos_info = []
        for modelo in modelos:
            arquivo = self.file_repo.get_by_id(db, modelo.file_id)
            
            modelo_info = InfoModelo(
                id=modelo.id,
                chave=modelo.key,
                nome_modelo=modelo.model,
                acuracia=modelo.accuracy,
                precisao=modelo.precision,
                recall=modelo.recall,
                f1_score=modelo.f1_score,
                roc_auc=modelo.roc_auc,
                estagio=modelo.stage.value,
                ativo=modelo.is_active,
                criado_em=modelo.created_at.isoformat(),
                arquivo_dataset=arquivo.name if arquivo else "N/A"
            )
            modelos_info.append(modelo_info)
        
        modelos_ativos = len([m for m in modelos if m.is_active])
        
        return ListaModelos(
            total=len(modelos),
            modelos=modelos_info
        )
    
    def promote_model(self, model_key: str, db: Session) -> RespostaPromocao:
        """
        Promote a model to champion.
        
        Args:
            model_key: Model key to be promoted
            db: Database session
            
        Returns:
            Promotion response
        """
        model_to_promote = self.model_repo.get_by_key(db, model_key)
        
        if not model_to_promote:
            raise ValueError(f"Model not found: {model_key}")
        
        if not model_to_promote.is_active:
            raise ValueError(f"Inactive model cannot be promoted: {model_key}")
        
        current_champion = self.model_repo.get_champion_model(db, model_to_promote.file_id)
        
        success = self.model_repo.promote_to_champion(db, model_key)
        
        if not success:
            raise ValueError("Failed to promote model")
        
        track_model_promotion(
            from_stage=current_champion.stage.value if current_champion else "none",
            to_stage="champion",
            model_name=model_to_promote.model
        )
        
        return RespostaPromocao(
            mensagem=f"Model {model_to_promote.model} successfully promoted to champion",
            modelo_promovido=model_key,
            modelo_anterior=current_champion.key if current_champion else None,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def archive_model(self, model_key: str, db: Session) -> bool:
        """
        Archive a model.
        
        Args:
            model_key: Model key
            db: Database session
            
        Returns:
            True if successfully archived
        """
        return self.model_repo.archive_model(db, model_key)
    
    def get_model_details(self, model_key: str, db: Session) -> Optional[DBResult]:
        """
        Get details of a specific model.
        
        Args:
            model_key: Model key
            db: Database session
            
        Returns:
            Model or None if not found
        """
        return self.model_repo.get_by_key(db, model_key)
    
    async def promote_model_with_validation(self, model_key: str, db: Session) -> RespostaPromocao:
        """Promotion with complete validation."""
        try:
            validation_result = await self._pre_promotion_validation(model_key, db)
            if not validation_result['valid']:
                raise ValueError(f"Validation failed: {validation_result['reason']}")
            
            impact = await self._assess_promotion_impact(model_key, db)
            if impact['risk_level'] > 0.7:
                raise ValueError(f"Risk too high for promotion: {impact['risk_level']}")
            
            backup = await self._create_promotion_backup(model_key, db)
            
            promotion_success = await self._safe_promotion(model_key, db, backup)
            
            if promotion_success:
                await self._start_observation_period(model_key, backup)
                
                return RespostaPromocao(
                    mensagem=f"Model successfully promoted (observation active)",
                    modelo_promovido=model_key,
                    impact_assessment=impact,
                    observacao_ativa=True,
                    timestamp=datetime.utcnow().isoformat()
                )
            else:
                raise ValueError("Failure during promotion")
                
        except Exception as e:
            await self._emergency_rollback_promotion(backup if 'backup' in locals() else None)
            raise

    async def _pre_promotion_validation(self, model_key: str, db: Session) -> Dict[str, Any]:
        """Validations before promotion."""
        model = self.model_repo.get_by_key(db, model_key)
        
        if not model:
            return {'valid': False, 'reason': 'Model not found'}
        
        if not model.is_active:
            return {'valid': False, 'reason': 'Inactive model'}
        
        if model.accuracy < 0.6:
            return {'valid': False, 'reason': f'Accuracy too low: {model.accuracy}'}
        
        try:
            Path(model.pickle_path).stat()
        except FileNotFoundError:
            return {'valid': False, 'reason': 'Model file not found'}
        
        model_age = (datetime.utcnow() - model.created_at).days
        if model_age > 30:
            return {'valid': False, 'reason': f'Model too old: {model_age} days'}
        
        return {'valid': True, 'reason': 'All validations passed'}

    async def _assess_promotion_impact(self, model_key: str, db: Session) -> Dict[str, Any]:
        """Assess promotion impact."""
        model = self.model_repo.get_by_key(db, model_key)
        current_champion = self.model_repo.get_champion_model(db, model.file_id)
        
        performance_delta = model.accuracy - (current_champion.accuracy if current_champion else 0)
        
        risk_factors = []
        
        if performance_delta < 0:
            risk_factors.append({'factor': 'performance_decrease', 'weight': 0.8})
        
        if not current_champion:
            risk_factors.append({'factor': 'no_current_champion', 'weight': 0.3})
        
        risk_level = min(sum(f['weight'] for f in risk_factors), 1.0)
        
        return {
            'performance_delta': performance_delta,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendation': 'PROCEED' if risk_level < 0.5 else 'CAUTION'
        }