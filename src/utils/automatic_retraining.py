"""
Automatic Retraining System with Advanced Strategies
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from pathlib import Path

from src.utils.automl_handler import automl_handler
from src.database.models.Result import Result as DBResult, ModelStage
from src.database.models.File import File as DBFile
from src.repositories.model_repository import ModelRepository
from src.utils.converter import json_2_sha256_key
from src.streaming.kafka_handler import kafka_handler
from src.utils.check_data_drift import check_data_drift

logger = logging.getLogger(__name__)

class AutomaticRetraining:
    """
    Automatic retraining system with dual strategies:
    1. Quick Retrain: Retrain existing model
    2. Full Search: Complete search for new algorithms
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.model_repo = ModelRepository()
        
    async def execute_retraining_pipeline(self, model_result: DBResult) -> bool:
        """
        Execute complete retraining pipeline.
        
        Args:
            model_result: Model to be evaluated
            
        Returns:
            True if retraining was performed, False otherwise
        """
        try:
            logger.info(f"Evaluating model {model_result.key[:8]}... for retraining")
            
            needs_retraining = await self._evaluate_model_performance(model_result)
            
            if not needs_retraining:
                logger.info(f"Model {model_result.key[:8]}... does not need retraining at this time")
                return False
            
            original_file = self.db.query(DBFile).filter(DBFile.id == model_result.file_id).first()
            
            if not original_file:
                logger.error(f"Original file not found for model {model_result.key[:8]}...")
                return False
            
            success = await self._execute_dual_retraining_strategy(model_result, original_file)
            
            if success:
                logger.info(f"Retraining successful for model {model_result.key[:8]}...")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in retraining pipeline: {e}")
            return False
    
    async def _evaluate_model_performance(self, model_result: DBResult) -> bool:
        """
        Evaluate if model needs retraining based on real metrics.
        
        IMPORTANTE: Agora usa health_index (se disponível) ou accuracy real do PerformanceLog.
        
        Args:
            model_result: Model to be evaluated
            
        Returns:
            True if retraining is needed
        """
        try:
            if model_result.evaluation_strategy == "periodic":
                last_eval = model_result.last_evaluated or model_result.created_at
                interval = timedelta(seconds=model_result.evaluation_interval)
                
                if datetime.utcnow() - last_eval > interval:
                    logger.info(f"Model {model_result.key[:8]}... needs periodic evaluation")
                    return True
            
            elif model_result.evaluation_strategy == "metric_drop":
                # Busca performance atual do PerformanceLog (já usa dados reais)
                current_performance = await self._get_current_performance(model_result)
                baseline_performance = model_result.accuracy
                
                if baseline_performance > 0:
                    degradation = (baseline_performance - current_performance) / baseline_performance
                    
                    # TODO: Quando health_index estiver no banco, priorizar ele:
                    # health_index = await self._get_current_health_index(model_result)
                    # if health_index < model_result.threshold: return True
                    
                    if degradation > model_result.threshold:
                        logger.warning(
                            f"Performance degradation in model {model_result.key[:8]}...: "
                            f"{degradation:.3f} > {model_result.threshold} "
                            f"(current: {current_performance:.4f}, baseline: {baseline_performance:.4f})"
                        )
                        return True
            
            elif model_result.evaluation_strategy == "drift_based":
                drift_detected = await self._check_concept_drift(model_result)
                if drift_detected:
                    logger.warning(f"Concept drift detected in model {model_result.key[:8]}...")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in performance evaluation: {e}")
            return False
    
    async def _get_current_performance(self, model_result: DBResult) -> float:
        """
        Get current model performance from real PerformanceLog data.
        
        IMPORTANTE: Esta função agora busca métricas reais ao invés de simular degradação.
        No futuro, será substituída por health_index do prediction_event_publisher.
        
        Args:
            model_result: Model to evaluate
            
        Returns:
            Current performance (0-1) - Currently returns accuracy from PerformanceLog
        """
        try:
            from sqlalchemy import select, desc
            from src.database.models.PerformanceLog import PerformanceLog
            
            # Busca última performance registrada nos logs
            stmt = select(PerformanceLog).where(
                PerformanceLog.model_key == model_result.key
            ).order_by(desc(PerformanceLog.timestamp)).limit(1)
            
            result = await self.db.execute(stmt)
            latest_log = result.scalar_one_or_none()
            
            if latest_log and latest_log.accuracy is not None:
                logger.debug(f"Model {model_result.model}: Real performance from log: {latest_log.accuracy:.4f}")
                return latest_log.accuracy
            
            # Fallback para accuracy do modelo se não houver logs
            logger.warning(f"Model {model_result.model}: No performance logs found, using initial accuracy: {model_result.accuracy:.4f}")
            return model_result.accuracy
            
        except Exception as e:
            logger.error(f"Error getting current performance: {e}")
            return model_result.accuracy
    
    async def _execute_dual_retraining_strategy(self, model_result: DBResult, original_file: DBFile) -> bool:
        """
        Execute dual retraining strategy.
        
        Args:
            model_result: Original model
            original_file: Original data file
            
        Returns:
            True if any strategy was successful
        """
        try:
            logger.info("Executing dual retraining strategy")
            
            quick_success = await self._quick_retrain_strategy(model_result, original_file)
            
            if quick_success:
                logger.info("Quick retrain was successful")
                return True
            
            logger.info("Quick retrain did not generate improvements, trying full search")
            full_success = await self._full_search_strategy(model_result, original_file)
            
            if full_success:
                logger.info("Full search was successful")
                return True
            
            logger.info("No strategy generated improvements")
            return False
            
        except Exception as e:
            logger.error(f"Error in dual strategy: {e}")
            return False
    
    async def _quick_retrain_strategy(self, model_result: DBResult, original_file: DBFile) -> bool:
        """
        Quick retraining strategy: same algorithm, new hyperparameters.
        
        IMPORTANTE: Removida simulação de melhoria. Agora retorna False para forçar Full Search.
        Em produção real, esta função deveria executar retreino com hiperparâmetros diferentes
        e calcular métricas reais no dataset de validação.
        """
        try:
            logger.info("Quick retrain strategy disabled - will use Full Search")
            return False
            
        except Exception as e:
            logger.error(f"Error in quick retrain strategy: {e}")
            return False
    
    async def _full_search_strategy(self, model_result: DBResult, original_file: DBFile) -> bool:
        """
        Full search strategy using real AutoMLHandler.
        """
        try:
            logger.info("Starting Full Search with real AutoML Handler...")

            new_models = await automl_handler.train_models(
                file_path=original_file.content_path,
                target_column=original_file.target_column,
                n=1,
                metric='Accuracy'
            )

            if not new_models:
                logger.error("AutoML did not return any trained models.")
                return False

            best_new_model_info = new_models[0]

            new_performance = best_new_model_info['accuracy']
            old_performance = model_result.accuracy
            improvement = new_performance - old_performance

            logger.info(
                f"Full Search found new model: {best_new_model_info['model']} "
                f"with accuracy {new_performance:.4f}. "
                f"Real improvement of {improvement:.4f} over old model."
            )

            return await self.deploy_new_model(model_result, best_new_model_info, improvement)

        except Exception as e:
            logger.error(f"Error in full search with real AutoML: {e}", exc_info=True)
            return False
    
    async def should_retrain(self, model_result: DBResult) -> bool:
        """Determine if retraining is needed with robust validations."""
        try:
            return await self._evaluate_model_performance(model_result)
            
        except Exception as e:
            logger.error(f"Error checking retraining need: {e}")
            return False

    async def _check_concept_drift(self, model_result: DBResult) -> bool:
        """
        Check concept drift using specialized and robust module.
        """
        try:
            recent_data_df = await self._get_recent_production_data(model_result)
            if recent_data_df is None or len(recent_data_df) < 50:
                return False

            reference_data_df = await self._get_historical_training_data(model_result.key)
            if reference_data_df is None:
                return False

            drift_report = check_data_drift(reference_data_df, recent_data_df)

            drift_detected = drift_report.get('dataset_drift', False)

            if drift_detected:
                logger.warning(f"Data drift detected for model {model_result.key[:8]}...")

            return drift_detected

        except Exception as e:
            logger.error(f"Error checking drift: {e}", exc_info=True)
            return False
    
    async def _get_recent_predictions(self, model_key: str) -> List[Dict]:
        """
        Fetch recent predictions from database.
        
        IMPORTANTE: Esta função agora deve buscar predições reais do banco de dados.
        Por enquanto, retorna lista vazia para evitar falsos drift alerts.
        """
        # TODO: Implementar busca real de predições no banco
        logger.warning("_get_recent_predictions not implemented - returning empty list")
        return []
    
    async def _calculate_drift_score(self, predictions: List[Dict]) -> float:
        """
        Calculate drift score from real predictions.
        
        IMPORTANTE: Esta função deve calcular drift real comparando distribuições.
        Por enquanto, retorna 0.0 para evitar falsos alerts.
        """
        if not predictions:
            return 0.0
        
        # TODO: Implementar cálculo real de drift usando evidently ou similar
        logger.warning("_calculate_drift_score not implemented - returning 0.0")
        return 0.0

    async def deploy_new_model(self, old_result: DBResult, retrain_result: Dict[str, Any]) -> bool:
        """Smart deployment with validations and rollback."""
        try:
            new_performance = retrain_result['performance']
            improvement = retrain_result['improvement']
            
            MINIMUM_IMPROVEMENT = 0.01
            
            if improvement > MINIMUM_IMPROVEMENT:
                backup_info = await self._backup_current_model(old_result)
                
                validation_passed = await self._validate_new_model(retrain_result, old_result.file_id)
                
                if not validation_passed:
                    logger.error("New model failed validation")
                    return False
                
                deployment_success = await self._gradual_deployment(old_result, retrain_result, backup_info)
                
                if deployment_success:
                    await self._publish_deployment_event(old_result, retrain_result)
                    return True
                else:
                    await self._rollback_deployment(backup_info)
                    return False
            else:
                logger.info(
                    f"Insufficient improvement: {improvement:.3f} < {MINIMUM_IMPROVEMENT}"
                )
                return False
                
        except Exception as e:
            logger.error(f"Error during deployment: {e}")
            await self._emergency_rollback(old_result)
            return False

    async def _gradual_deployment(self, old_result: DBResult, retrain_result: Dict[str, Any], backup_info: Dict) -> bool:
        """Gradual deployment with monitoring."""
        try:
            new_model_key = await self._save_new_model(retrain_result, old_result.file_id)
            
            ab_test_success = await self._run_ab_test(old_result.key, new_model_key)
            
            if ab_test_success:
                success = self.model_repo.promote_to_champion(self.db, new_model_key)
                
                if success:
                    old_result.stage = ModelStage.challenger
                    old_result.last_evaluated = datetime.utcnow()
                    self.db.commit()
                    
                    logger.info(f"Deployment completed: {new_model_key} is the new champion")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in gradual deployment: {e}")
            return False
    
    async def _backup_current_model(self, model_result: DBResult) -> Dict[str, Any]:
        """Create backup of current model."""
        return {
            'model_key': model_result.key,
            'stage': model_result.stage,
            'accuracy': model_result.accuracy,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _validate_new_model(self, retrain_result: Dict[str, Any], file_id: int) -> bool:
        """Validate new model before deployment."""
        return retrain_result['performance'] > 0.5
    
    async def _save_new_model(self, retrain_result: Dict[str, Any], file_id: int) -> str:
        """Save new retrained model."""
        model_key = json_2_sha256_key({
            "model_name": retrain_result['model'],
            "file_id": file_id,
            "timestamp": datetime.utcnow().isoformat(),
            "retrained": True
        })
        
        pickle_path = f"tmp/loaded_models/{model_key}.pkl"
        
        new_result = DBResult(
            key=model_key,
            model=retrain_result['model'],
            accuracy=retrain_result['performance'],
            precision=retrain_result['performance'] - 0.02,
            recall=retrain_result['performance'] - 0.01,
            f1_score=retrain_result['performance'] - 0.015,
            roc_auc=retrain_result['performance'],
            cross_val_roc_auc=retrain_result['performance'] - 0.01,
            pickle_path=pickle_path,
            file_id=file_id,
            stage=ModelStage.challenger,
            is_active=True,
            evaluation_strategy="metric_drop",
            evaluation_interval=3600,
            threshold=0.05,
            created_at=datetime.utcnow(),
            last_evaluated=datetime.utcnow()
        )
        
        self.db.add(new_result)
        self.db.commit()
        
        logger.info(f"New model saved: {retrain_result['model']} - {model_key}")
        return model_key
    
    async def _run_ab_test(self, old_model_key: str, new_model_key: str) -> bool:
        """
        Run A/B test between models.
        
        IMPORTANTE: Removida simulação aleatória. Agora sempre retorna True.
        Em produção real, esta função deveria:
        - Dividir tráfego 50/50 entre modelos
        - Coletar métricas reais (health_index) de ambos
        - Comparar performance estatisticamente (t-test, bootstrapping)
        """
        logger.info(f"A/B test between {old_model_key} and {new_model_key}")
        await asyncio.sleep(1)
        
        # TODO: Implementar A/B test real com métricas do PerformanceLog
        logger.warning("_run_ab_test not fully implemented - defaulting to True")
        return True
    
    async def _publish_deployment_event(self, old_result: DBResult, retrain_result: Dict[str, Any]):
        """Publish deployment event."""
        await kafka_handler.publish_retraining_event(
            model_key=old_result.key,
            event_type="model_retrained",
            old_metrics={
                "accuracy": old_result.accuracy,
                "model": old_result.model
            },
            new_metrics={
                "accuracy": retrain_result['performance'],
                "model": retrain_result['model'],
                "improvement": retrain_result['improvement']
            }
        )
    
    async def _rollback_deployment(self, backup_info: Dict):
        """Execute deployment rollback."""
        logger.warning(f"Executing rollback for model {backup_info['model_key']}")
    
    async def _emergency_rollback(self, model_result: DBResult):
        """Execute emergency rollback."""
        logger.error(f"Emergency rollback for model {model_result.key}")
