"""
Prediction Service
"""
import logging
import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from cachetools import TTLCache

from src.database.config import SessionLocal
from src.database.models.Result import Result as DBResult
from src.utils.prediction_event_publisher import publicador_predicao
from src.utils.data_utils import remove_target_columns, get_feature_columns

logger = logging.getLogger(__name__)

class PredictionService:
    """Service for model predictions"""
    
    def __init__(self):
        self.loaded_models = TTLCache(maxsize=100, ttl=3600)  # Cache com TTL de 1 hora
    
    def _load_model(self, model_path: str):
        """
        Load model from file with cache.

        Args:
            model_path: Model file path.

        Returns:
            Loaded model or None.
        """
        if model_path in self.loaded_models:
            return self.loaded_models[model_path]
        
        try:
            # Use Path to handle cross-platform paths correctly
            model_path_obj = Path(model_path)
            
            # Make path absolute if it's relative
            if not model_path_obj.is_absolute():
                model_path_obj = Path.cwd() / model_path_obj
            
            logger.info(f"[MODEL LOAD] Original path: {model_path}")
            logger.info(f"[MODEL LOAD] Resolved path: {model_path_obj}")
            logger.info(f"[MODEL LOAD] Path exists: {model_path_obj.exists()}")
            
            if not model_path_obj.exists():
                logger.error(f"[MODEL LOAD] Model file not found at: {model_path_obj}")
                return None
            
            model = joblib.load(model_path_obj)
            if hasattr(model, 'predict'):
                self.loaded_models[model_path] = model
                logger.info(f"[MODEL LOAD] Model loaded successfully and cached")
                return model
            else:
                logger.error(f"[MODEL LOAD] Loaded object does not have predict method")
                return None
        except Exception as e:
            logger.error(f"Model loading error: {e}", exc_info=True)
        return None
    
    async def predict_batch(self, model_key: str, data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform batch predictions.

        Args:
            model_key: Model key.
            data_list: List of input data.

        Returns:
            Prediction results.
        """
        logger.info(f"[SERVICE] predict_batch called")
        logger.info(f"[SERVICE] model_key: {model_key[:16]}...")
        logger.info(f"[SERVICE] data_list type: {type(data_list)}")
        logger.info(f"[SERVICE] data_list size: {len(data_list) if isinstance(data_list, list) else 'NOT A LIST!'}")
        
        db: Session = SessionLocal()
        
        try:
            logger.info(f"[SERVICE] Fetching model from database...")
            model_result = db.query(DBResult).filter(
                DBResult.key == model_key,
                DBResult.is_active == True
            ).first()
            
            if not model_result:
                logger.error(f"[SERVICE] Model not found: {model_key}")
                raise ValueError(f"Model not found: {model_key}")
            
            logger.info(f"[SERVICE] Model found: {model_result.model}")
            logger.info(f"[SERVICE] Loading model from: {model_result.pickle_path}")
            
            model = self._load_model(model_result.pickle_path)
            if not model:
                logger.error(f"[SERVICE] Failed to load model")
                raise ValueError("Model loading failed")
            
            logger.info(f"[SERVICE] Model loaded successfully")
            
            # Feature preparation
            logger.info(f"[FEATURES] Getting expected columns from model...")
            expected_columns_raw = list(model.feature_names_in_)
            logger.info(f"[FEATURES] Model expects {len(expected_columns_raw)} columns: {expected_columns_raw}")
            
            # CRITICAL FIX: Detect data leakage - remove target columns from model's expected features
            # Models were trained incorrectly with target column (CO2_daily_kg) as a feature
            from src.utils.data_utils import DEFAULT_TARGET_COLUMNS
            expected_columns = [col for col in expected_columns_raw if col not in DEFAULT_TARGET_COLUMNS]
            
            if len(expected_columns) < len(expected_columns_raw):
                removed_targets = [col for col in expected_columns_raw if col in DEFAULT_TARGET_COLUMNS]
                logger.warning(f"[DATA LEAKAGE FIX] Model was trained with target columns as features!")
                logger.warning(f"[DATA LEAKAGE FIX] Removing from expected features: {removed_targets}")
                logger.warning(f"[DATA LEAKAGE FIX] Corrected features: {expected_columns}")
            
            logger.info(f"[FEATURES] Creating DataFrame from data_list...")
            df = pd.DataFrame(data_list)
            logger.info(f"[FEATURES] DataFrame created - Shape: {df.shape}, Columns: {list(df.columns)}")
            
            # Remove target columns (CO2_daily_kg, CO2_category, etc)
            logger.info(f"[FEATURES] Removing target columns...")
            df = remove_target_columns(df)
            logger.info(f"[FEATURES] After removing targets - Columns: {list(df.columns)}")
            
            # Ensure we have ONLY the columns expected by the model (after removing targets)
            logger.info(f"[FEATURES] Checking feature availability...")
            available_columns = [col for col in expected_columns if col in df.columns]
            missing_columns = [col for col in expected_columns if col not in df.columns]
            
            logger.info(f"[FEATURES] Available features: {len(available_columns)}/{len(expected_columns)}")
            if missing_columns:
                logger.error(f"[FEATURES] Missing features: {missing_columns}")
                raise ValueError(f"Missing required features: {missing_columns}")
            
            # Select only the columns in correct order
            logger.info(f"[FEATURES] Selecting columns in correct order...")
            df = df[expected_columns]
            logger.info(f"[FEATURES] Final DataFrame - Shape: {df.shape}")
            
            logger.info(f"[PREDICTION] Executing prediction...")
            predictions = model.predict(df)
            logger.info(f"[PREDICTION] Complete - {len(predictions)} results")
            
            # Detect model type (classification vs regression)
            logger.info(f"[TYPE] Detecting model type...")
            is_classification = hasattr(model, 'predict_proba')
            logger.info(f"[TYPE] Model is classification: {is_classification}")
            
            probabilities = None
            pseudo_confidences = None
            
            if is_classification:
                # CLASSIFICATION: Use predict_proba
                probabilities = model.predict_proba(df).tolist()
            else:
                # REGRESSION: Calculate pseudo-confidence based on variance
                # We use prediction variance as inverse confidence proxy
                # Lower variance = higher confidence
                pred_array = np.array(predictions)
                pred_std = np.std(pred_array) if len(pred_array) > 1 else 0.0
                
                # Normalize confidence: typical values have std between 0-10
                # Confidence = 1 / (1 + std_normalized)
                # std = 0 → confidence = 1.0
                # std = 5 → confidence = 0.67
                # std = 10 → confidence = 0.50
                max_std = max(pred_std, 0.01)  # Avoid division by zero
                pseudo_confidences = [1.0 / (1.0 + (pred_std / 10.0))] * len(predictions)
                
                logger.debug(f"Regression model: std={pred_std:.3f}, pseudo_confidence={pseudo_confidences[0]:.3f}")
            
            results = []
            for i, prediction in enumerate(predictions):
                result_dict = {
                    'prediction': float(prediction) if isinstance(prediction, (np.integer, np.floating)) else prediction
                }
                
                if is_classification:
                    prob = probabilities[i]
                    confidence = max(prob)
                    result_dict['confidence'] = confidence
                    result_dict['probabilities'] = prob
                else:
                    # Regression: only include confidence, no probabilities
                    confidence = pseudo_confidences[i] if pseudo_confidences else None
                    if confidence is not None:
                        result_dict['confidence'] = confidence
                
                results.append(result_dict)
            
            # Capture health_index from prediction_event_publisher
            health_index = await publicador_predicao.process_prediction_batch(
                model_key=model_key,
                predictions=predictions.tolist(),
                probabilities=probabilities if probabilities else [],
                pseudo_confidences=pseudo_confidences if pseudo_confidences else [],
                is_classification=is_classification,
                input_data=df,
                db_session=db
            )
            
            return {
                'predictions': results,
                'model_key': model_key,
                'model_name': model_result.model,
                'timestamp': pd.Timestamp.now().isoformat(),
                'total_predictions': len(results),
                'health_index': health_index  # Add health_index to result
            }
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise
        finally:
            db.close()

prediction_service = PredictionService()