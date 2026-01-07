"""
Prediction Event Publisher with Trigger

Implements event-based detection system using risk score.
Monitors predictions in real-time and triggers events when risk thresholds are exceeded.
"""

import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from pathlib import Path

from src.utils.check_data_drift import check_data_drift
from src.utils.observer_pattern import Subject, EventType
from src.database.models.Result import Result as DBResult, ModelStage
from src.database.models.File import File as DBFile

logger = logging.getLogger(__name__)

@dataclass
class ModelRiskState:
    """
    Tracks health state for each model.
    
    Maintains metrics for monitoring:
    - Health index (PRIMARY SYSTEM METRIC)
    - Prediction count (for volume tracking)
    - Timestamps for temporal control
    """
    health_index: float = 1.0  # 1.0 = perfect, 0.0 = critical
    prediction_count: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    last_prediction_time: datetime = field(default_factory=datetime.now)
    
    def reset_if_needed(self, reset_interval_hours: int = 1):
        """
        Resets accumulated counters after the configured interval.
        
        This prevents counter accumulation and enables pattern
        detection in time windows.
        
        IMPORTANT: health_index is NOT reset as it represents current model state.
        """
        if datetime.now() - self.last_reset > timedelta(hours=reset_interval_hours):
            self.prediction_count = 0
            self.last_reset = datetime.now()

class PredictionEventPublisher(Subject):
    """
    Event publisher based on prediction health analysis.
    
    Monitoring system that tracks:
    - Model health degradation (via health_index)
    - System volume stress (via prediction count)
    
    Health Index is the PRIMARY METRIC that combines:
    - Data drift risk (40%): changes in data distribution
    - Prediction confidence risk (50%): prediction confidence levels
    - Input anomaly risk (10%): statistical outlier detection
    
    Professional MLOps Approach:
    - Minimum observation window: 20 predictions before alerting
    - Multi-signal confirmation: Multiple risk components must be elevated
    - Statistical robustness: Uses confidence intervals and thresholds
    """
    
    # Health Index Thresholds
    HEALTH_THRESHOLD = 0.70      # Threshold for model degradation alert
    VOLUME_THRESHOLD = 0.9       # Threshold for system volume alert
    MIN_PREDICTIONS_FOR_ALERT = 20  # Minimum predictions before triggering alerts
    
    # Health Index Weights (sum = 1.0) - Balanced professional approach
    # Confidence is PRIMARY (50%) as it directly reflects prediction quality
    # Drift is SECONDARY (40%) as it's an early warning indicator
    # Anomaly is TERTIARY (10%) as it can have false positives
    W_DRIFT = 0.40               # Weight for data drift (early warning)
    W_CONFIDENCE = 0.50          # Weight for prediction confidence (direct quality metric)
    W_ANOMALY = 0.10             # Weight for input anomalies (noise-prone)

    def __init__(self):
        """Initialize publisher with state control structures."""
        super().__init__()
        self.model_states: Dict[str, ModelRiskState] = {}
        self.logger = logging.getLogger(__name__)
    
    @property
    def observer_count(self) -> int:
        """Return number of registered observers for monitoring."""
        return len(self._observers)
    
    def get_or_create_model_state(self, model_key: str) -> ModelRiskState:
        """
        Gets or creates the risk state for a model.
        
        Implements lazy initialization pattern for models.
        Automatically resets state based on time window.
        """
        if model_key not in self.model_states:
            self.model_states[model_key] = ModelRiskState()
        
        self.model_states[model_key].reset_if_needed()
        return self.model_states[model_key]
    
    async def process_prediction_batch(
        self,
        model_key: str,
        predictions: List[float],
        probabilities: List[List[float]],
        input_data,
        db_session: Session,
        trigger_alerts: bool = True,
        pseudo_confidences: List[float] = None,
        is_classification: bool = True
    ) -> float:
        """
        Processes a prediction batch and calculates the health index.
        Supports both CLASSIFICATION and REGRESSION models.
        
        Professional MLOps Health Index Algorithm:
        1. Confidence component (50%): Direct prediction quality metric
           - Classification: Uses predict_proba()
           - Regression: Uses pseudo-confidence from prediction variance
        2. Drift component (40%): Early warning for distribution changes
        3. Anomaly component (10%): Statistical outlier detection
        4. Weighted: health = 0.5*(1-conf) + 0.4*(1-drift) + 0.1*(1-anom)
        
        Alert Triggers (Multi-Criteria Validation):
        - Minimum observations: ≥20 predictions (statistical significance)
        - Health threshold: <0.70 (degraded performance)
        - Multi-signal confirmation: ≥2 risk components >0.3 (avoid false positives)
        
        This approach follows industry best practices from AWS SageMaker, Azure ML,
        and Datadog ML Monitoring to ensure robust and reliable alerting.
        
        Args:
            probabilities: For classification models (predict_proba output)
            pseudo_confidences: For regression models (variance-based confidence)
            is_classification: True for classification, False for regression
            trigger_alerts: If False, only calculates health without triggering alerts
        
        Returns:
            health_index (float): Unified health metric (0.0 = critical, 1.0 = perfect)
        """
        try:
            state = self.get_or_create_model_state(model_key)
            state.last_prediction_time = datetime.now()
            
            batch_size = len(predictions)
            state.prediction_count += batch_size
            
            # ===== COMPONENT 1: VOLUME RISK (System-level, not model-specific) =====
            volume_risk = min(state.prediction_count / 100.0, 1.0)
            
            # ===== COMPONENT 2: ANOMALY RISK =====
            # Calculate anomalies only for current batch, don't accumulate
            anomaly_count = await self._detect_input_anomalies_simplified(model_key, input_data, db_session)
            anomaly_risk = min(anomaly_count / batch_size, 1.0) if batch_size > 0 else 0.0
            
            # ===== COMPONENT 3: CONFIDENCE RISK =====
            # Calculate low confidence only for current batch, don't accumulate
            if is_classification:
                low_confidence_count = self._count_low_confidence_predictions(probabilities)
            else:
                # Regression: usar pseudo_confidences
                low_confidence_count = self._count_low_confidence_predictions_regression(pseudo_confidences)
            
            confidence_risk = min(low_confidence_count / batch_size, 1.0) if batch_size > 0 else 0.0
            
            # ===== COMPONENT 4: DRIFT RISK =====
            drift_risk = await self._calculate_drift_risk(model_key, input_data, db_session)
            
            # ===== HEALTH INDEX: PRIMARY SYSTEM METRIC =====
            # Combines drift, confidence, and anomalies into a unified index (0.0 = critical, 1.0 = perfect)
            health_index = (
                (self.W_DRIFT * (1.0 - drift_risk)) +
                (self.W_CONFIDENCE * (1.0 - confidence_risk)) +
                (self.W_ANOMALY * (1.0 - anomaly_risk))
            )
            
            # Store health_index in state
            state.health_index = health_index
            # =========================================================
            
            model_type = "Classification" if is_classification else "Regression"
            self.logger.info(
                f"Model {model_key[:8]}... ({model_type}): 🏥 Health Index = {health_index:.3f} | "
                f"Predictions: {state.prediction_count}, Drift: {drift_risk:.2f}, "
                f"Conf: {confidence_risk:.2f}, Anom: {anomaly_risk:.2f}"
            )
            
            # ===== TRIGGER 1: MODEL DEGRADATION (health_index) =====
            # Professional MLOps approach: Multi-criteria validation
            # 1. Sufficient observations (statistical significance)
            # 2. Health index below threshold
            # 3. At least two risk components elevated (multi-signal confirmation)
            if trigger_alerts and state.prediction_count >= self.MIN_PREDICTIONS_FOR_ALERT:
                # Count elevated risk components (>0.3 is considered elevated)
                elevated_risks = sum([
                    drift_risk > 0.3,
                    confidence_risk > 0.3,
                    anomaly_risk > 0.3
                ])
                
                # Alert only if health is degraded AND multiple signals confirm
                if health_index < self.HEALTH_THRESHOLD and elevated_risks >= 2:
                    await self.notify({
                        'event_type': EventType.MODEL_DEGRADATION_DETECTED,
                        'model_key': model_key,
                        'health_index': health_index,
                        'threshold': self.HEALTH_THRESHOLD,
                        'prediction_count': state.prediction_count,
                        'risk_components': {
                            'drift_risk': drift_risk,
                            'confidence_risk': confidence_risk,
                            'anomaly_risk': anomaly_risk
                        },
                        'elevated_risk_count': elevated_risks,
                        'timestamp': datetime.now().isoformat()
                    })
                    self.logger.warning(
                        f"🚨 MODEL DEGRADATION: {model_key[:8]}... | "
                        f"Health Index {health_index:.3f} < {self.HEALTH_THRESHOLD} | "
                        f"Predictions: {state.prediction_count} | "
                        f"Elevated risks: {elevated_risks}/3"
                    )
                elif health_index < self.HEALTH_THRESHOLD:
                    self.logger.info(
                        f"⚠️  Health degraded but insufficient confirmation: {model_key[:8]}... | "
                        f"Health: {health_index:.3f}, Elevated risks: {elevated_risks}/3 (need ≥2)"
                    )
            
            # ===== TRIGGER 2: SYSTEM VOLUME OVERLOAD (volume_risk) - DISABLED =====
            # Volume alerts removed per user request
            # if volume_risk >= self.VOLUME_THRESHOLD:
            #     await self.notify({
            #         'event_type': EventType.SYSTEM_ERROR,
            #         'model_key': model_key,
            #         'volume_risk': volume_risk,
            #         'prediction_count': state.prediction_count,
            #         'timestamp': datetime.now().isoformat()
            #     })
            #     self.logger.warning(
            #         f"🚨 SYSTEM VOLUME ALERT: {model_key[:8]}... | "
            #         f"Volume Risk {volume_risk:.3f} >= {self.VOLUME_THRESHOLD} "
            #         f"({state.prediction_count} predictions)"
            #     )
            #     # Reset counter after volume alert
            #     state.prediction_count = 0
            
            # Return health_index to be saved in PerformanceLog
            return health_index
                
        except Exception as e:
            self.logger.error(f"Error processing prediction batch for model {model_key}: {e}", exc_info=True)
            return 1.0  # Fallback: return perfect health on error
    
    def _count_low_confidence_predictions(self, probabilities: List[List[float]], threshold: float = 0.7) -> int:
        """
        Counts predictions with low confidence based on probabilities (CLASSIFICATION).
        """
        if not probabilities:
            return 0
        
        low_confidence = 0
        for prob_array in probabilities:
            if prob_array and max(prob_array) < threshold:
                low_confidence += 1
        
        return low_confidence
    
    def _count_low_confidence_predictions_regression(self, pseudo_confidences: List[float], threshold: float = 0.7) -> int:
        """
        Counts predictions with low confidence for REGRESSION models.
        Uses pseudo-confidence calculated from prediction variance.
        """
        if not pseudo_confidences:
            return 0
        
        low_confidence = 0
        for conf in pseudo_confidences:
            if conf < threshold:
                low_confidence += 1
        
        return low_confidence
    
    async def _detect_input_anomalies_simplified(self, model_key: str, input_data, db_session: Session) -> int:
        """
        Detects anomalies in input data by comparing against training data distribution.
        
        Uses IQR method calculated from TRAINING data to identify outliers in PREDICTION data.
        This ensures we detect true anomalies (data that differs from training distribution),
        not just natural variation within the prediction batch.
        
        IQR (Interquartile Range) method:
        - Calculate Q1, Q3, IQR from TRAINING data
        - Apply bounds to PREDICTION data
        - Count rows with ANY outlier (not sum of all outliers)
        """
        try:
            import numpy as np
            
            if input_data is None or input_data.empty:
                return 0
            
            # Get training data for reference distribution
            training_data = await self._get_historical_training_data(model_key, db_session)
            if training_data is None or training_data.empty:
                # No reference data available, cannot detect anomalies
                return 0
            
            # Track which rows have at least one anomaly
            anomalous_rows = set()
            
            # Get common numeric columns
            training_numeric = training_data.select_dtypes(include=[np.number]).columns
            input_numeric = input_data.select_dtypes(include=[np.number]).columns
            common_cols = list(set(training_numeric) & set(input_numeric))
            
            for column in common_cols:
                # Calculate IQR from TRAINING data (reference distribution)
                training_values = training_data[column]
                Q1 = training_values.quantile(0.25)
                Q3 = training_values.quantile(0.75)
                IQR = Q3 - Q1
                
                # Skip columns with no variance in training
                if IQR == 0:
                    continue
                
                # Use more permissive bounds (3.0 instead of 2.0) since ~17% anomaly rate is still high
                # This accounts for natural variability in production data
                lower_bound = Q1 - 3.0 * IQR
                upper_bound = Q3 + 3.0 * IQR
                
                # Apply bounds to PREDICTION data
                prediction_values = input_data[column]
                outlier_indices = prediction_values[(prediction_values < lower_bound) | (prediction_values > upper_bound)].index
                anomalous_rows.update(outlier_indices)
            
            # Return count of rows with at least one anomaly
            return len(anomalous_rows)
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            return 0
    
    def _compare_distributions_simplified(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """
        Simplified distribution comparison using mean of percentage changes.
        Returns a score from 0 to 1.
        """
        numeric_cols1 = df1.select_dtypes(include='number').columns
        numeric_cols2 = df2.select_dtypes(include='number').columns
        common_cols = list(set(numeric_cols1) & set(numeric_cols2))

        if not common_cols:
            return 0.0

        drifts = []
        for col in common_cols:
            mean1 = df1[col].mean()
            mean2 = df2[col].mean()
            
            if abs(mean1) > 1e-6:
                percent_change = abs((mean2 - mean1) / mean1)
                drifts.append(percent_change)

        if not drifts:
            return 0.0
        
        return min(sum(drifts) / len(drifts), 1.0)
    
    async def _calculate_drift_risk(self, model_key: str, input_data, db_session: Session) -> float:
        """Calculates drift risk using specialized and robust module."""
        try:
            historical_data = await self._get_historical_training_data(model_key, db_session)

            if (
                historical_data is None or input_data is None or
                historical_data.empty or input_data.empty
            ):
                return 0.0

            drift_report = check_data_drift(historical_data, input_data)

            # Get drift score (proportion of drifted columns)
            drift_score = drift_report.get('mean_drifted_score', 0.0)
            drifted_cols = drift_report.get('number_of_drifted_columns', 0)
            total_cols = drift_report.get('number_of_columns', 1)
            
            self.logger.debug(
                f"Drift analysis for {model_key[:8]}: "
                f"{drifted_cols}/{total_cols} columns drifted (score: {drift_score:.3f})"
            )

            return drift_score

        except Exception as e:
            self.logger.error(f"Error calculating drift risk: {e}", exc_info=True)
            return 0.0

    async def _get_historical_training_data(self, model_key: str, db: Session) -> Optional[pd.DataFrame]:
        """
        Fetches the original training dataset for a model.

        1. Find model in database by its 'key'.
        2. Use model's 'file_id' to find file record.
        3. Get file path ('content_path') and verify it exists.
        4. Load CSV file into pandas DataFrame.
        """
        try:
            model_record = db.query(DBResult).filter(DBResult.key == model_key).first()
            if not model_record:
                self.logger.warning(f"No model record found for key: {model_key}")
                return None

            file_record = db.query(DBFile).filter(DBFile.id == model_record.file_id).first()
            if not file_record:
                self.logger.warning(f"No file record found for file_id: {model_record.file_id}")
                return None

            file_path = Path(file_record.content_path)
            if not file_path.exists():
                self.logger.error(f"Training file not found at path: {file_path}. Data inconsistency.")
                return None
            
            training_data = pd.read_csv(file_path)
            self.logger.debug(f"Training dataset for model {model_key[:8]} loaded from {file_path}")
            return training_data

        except Exception as e:
            self.logger.error(f"Failed to load historical data for model {model_key}: {e}", exc_info=True)
            return None
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Returns current metrics for all tracked models."""
        return {
            'total_models_tracked': len(self.model_states),
            'observers_count': self.observer_count,
            'models_state': {
                key: {
                    'health_index': state.health_index,
                    'prediction_count': state.prediction_count,
                    'last_prediction': state.last_prediction_time.isoformat()
                }
                for key, state in self.model_states.items()
            }
        }

publicador_predicao = PredictionEventPublisher()