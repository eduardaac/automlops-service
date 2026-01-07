"""
AutoML Handler using PyCaret - Version with Governance and Validation
Centralizes all automatic ML model training logic
"""
import logging
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

import mlflow
from src.utils.converter import json_2_sha256_key
from src.utils.data_validator import data_validator

logger = logging.getLogger(__name__)

# MLflow URI - detecta automaticamente se está em Docker ou local
import os
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
logger.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")


class AutoMLHandler:
    """Handler for AutoML operations with PyCaret, with governance via MLflow."""
    
    async def train_models(self, 
                          file_path: str, 
                          target_column: str, 
                          n: int = 1,
                          metric: str = 'Accuracy',
                          model_types: List[str] = None,
                          **kwargs) -> List[Dict[str, Any]]:
        """
        Train models using AutoML with data validation and experiment tracking.
        
        Args:
            file_path: Path to training dataset CSV
            target_column: Name of target column
            n: Number of models to train
            metric: Optimization metric
            model_types: List of specific model types to train (PyCaret codes)
                        e.g., ['rf', 'et', 'lightgbm', 'gbc']
                        If None, uses default model list
            **kwargs: Additional parameters for PyCaret setup
        
        Returns:
            List of trained model information dictionaries
        """
        experiment_name = f"Training_{Path(file_path).stem}"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"AutoML_Run_{datetime.now().strftime('%Y%m%d-%H%M%S')}") as parent_run:
            try:
                logger.info(f"Starting AutoML training with MLflow tracking (Run ID: {parent_run.info.run_id})")

                mlflow.log_params({
                    "file_path": file_path,
                    "target_column": target_column,
                    "n_models_requested": n,
                    "optimization_metric": metric
                })

                dataset = pd.read_csv(file_path)
                logger.info(f"Dataset loaded: {dataset.shape}")
                
                feature_df = dataset.drop(columns=[target_column], errors='ignore')
                # Validação flexível - aceita qualquer schema de dados
                if not data_validator.validate_dataframe(feature_df, min_rows=10, check_nulls=False):
                    raise ValueError(f"Input data validation failed: {data_validator.validation_errors}")
                logger.info("Dataset data validation completed successfully.")

                problema_tipo = self._detect_problem_type(dataset, target_column)
                metric = self._adjust_metric_for_problem_type(problema_tipo, metric)
                pycaret_kwargs, custom_kwargs = self._separate_kwargs(kwargs)
                
                await self._setup_ml_environment(dataset, target_column, problema_tipo, **pycaret_kwargs)
                comparison_df = await self._compare_models(n, metric, problema_tipo, model_types)
                
                modelos_finalizados = await self._create_final_models(
                    comparison_df, dataset, target_column, n, custom_kwargs, problema_tipo
                )
                
                mlflow.set_tag("status", "COMPLETED")
                logger.info(f"Training completed: {len(modelos_finalizados)} models created and registered in MLflow.")
                return modelos_finalizados
            
            except Exception as e:
                logger.error(f"Critical error in AutoML training: {e}", exc_info=True)
                mlflow.set_tag("status", "FAILED")
                mlflow.log_param("error_message", str(e))
                raise

    def _detect_problem_type(self, dataset: pd.DataFrame, target_column: str) -> str:
        target_series = dataset[target_column]
        unique_values = target_series.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(target_series)
        
        if not is_numeric or unique_values <= 10 or target_series.dtype == 'bool':
            problem_type = 'classification'
        else:
            problem_type = 'regression'
        
        logger.info(f"Problem type detected: {problem_type}")
        logger.info(f"  - Target column: {target_column}")
        logger.info(f"  - Is numeric: {is_numeric}")
        logger.info(f"  - Unique values: {unique_values}")
        logger.info(f"  - Data type: {target_series.dtype}")
        
        return problem_type

    async def _setup_ml_environment(self, 
                                   dataset: pd.DataFrame, 
                                   target_column: str,
                                   problem_type: str,
                                   **pycaret_kwargs) -> Any:
        try:
            if problem_type == 'classification':
                from pycaret.classification import setup
            else:
                from pycaret.regression import setup
            
            # CRITICAL FIX: Ensure target column exists and is valid
            if target_column not in dataset.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset. Available: {list(dataset.columns)}")
            
            # CRITICAL FIX: AGGRESSIVE data leakage prevention
            # Remove ALL other target columns from dataset BEFORE passing to PyCaret
            from src.utils.data_utils import DEFAULT_TARGET_COLUMNS
            
            # Step 1: Identify other target columns to remove
            other_target_columns = [
                col for col in DEFAULT_TARGET_COLUMNS 
                if col in dataset.columns and col != target_column
            ]
            
            if other_target_columns:
                logger.warning(f"⚠️ REMOVING other target columns to prevent leakage: {other_target_columns}")
                dataset = dataset.drop(columns=other_target_columns, errors='ignore')
            
            # Step 2: Get final feature columns (everything except target)
            feature_columns = [col for col in dataset.columns if col != target_column]
            
            # Step 3: Log setup information
            logger.info(f"=" * 80)
            logger.info(f"🔧 PYCARET SETUP - DATA LEAKAGE PREVENTION")
            logger.info(f"=" * 80)
            logger.info(f"Dataset shape: {dataset.shape}")
            logger.info(f"Target column: '{target_column}'")
            logger.info(f"Feature columns ({len(feature_columns)}): {feature_columns}")
            logger.info(f"Removed columns: {other_target_columns if other_target_columns else 'None'}")
            logger.info(f"=" * 80)
            
            # Step 4: CRITICAL - Create separate X and y, then recombine
            # This ensures PyCaret CANNOT use target as a feature
            X = dataset[feature_columns].copy()
            y = dataset[target_column].copy()
            
            # Recombine with target at the end
            dataset_final = X.copy()
            dataset_final[target_column] = y
            
            logger.info(f"✅ Final dataset for PyCaret:")
            logger.info(f"   - Shape: {dataset_final.shape}")
            logger.info(f"   - Columns: {list(dataset_final.columns)}")
            logger.info(f"   - Target is LAST column: {dataset_final.columns[-1] == target_column}")
            
            # Step 5: Setup PyCaret with clean dataset
            default_setup_config = {
                'data': dataset_final,
                'target': target_column,
                'session_id': 123,
                'verbose': False,
                'html': False,   # Disable HTML output
                'ignore_features': None  # CRITICAL: Force PyCaret to use ONLY our features
            }
            setup_config = {**default_setup_config, **pycaret_kwargs}
            
            # Remove None values from config
            setup_config = {k: v for k, v in setup_config.items() if v is not None}
            
            # CRITICAL VALIDATION: Ensure target is NOT in features before setup
            all_columns = list(dataset_final.columns)
            if target_column in all_columns[:-1]:  # Check all except last column
                raise ValueError(f"CRITICAL BUG: Target '{target_column}' is not the last column! Order: {all_columns}")
            
            logger.info(f"🔒 FINAL VALIDATION BEFORE PYCARET:")
            logger.info(f"   - Total columns: {len(all_columns)}")
            logger.info(f"   - Last column (target): '{all_columns[-1]}'")
            logger.info(f"   - Features (n={len(all_columns)-1}): {all_columns[:-1]}")
            
            logger.info("⏳ Calling PyCaret setup()...")
            setup_result = setup(**setup_config)
            logger.info("✅ PyCaret setup() completed")
            
            # POST-SETUP VALIDATION: Verify no data leakage occurred
            logger.info("=" * 80)
            logger.info("🔍 POST-SETUP VALIDATION - CHECKING FOR DATA LEAKAGE")
            logger.info("=" * 80)
            
            try:
                if problem_type == 'classification':
                    from pycaret.classification import get_config
                else:
                    from pycaret.regression import get_config
                
                logger.info("📊 Retrieving X_train from PyCaret...")
                X_train = get_config('X_train')
                actual_features = list(X_train.columns)
                expected_features = feature_columns
                
                logger.info(f"✓ Expected features (n={len(expected_features)}): {expected_features}")
                logger.info(f"✓ Actual features in X_train (n={len(actual_features)}): {actual_features}")
                
                # CRITICAL CHECK: Target must NOT be in X_train
                if target_column in actual_features:
                    error_msg = (
                        f"\n{'='*80}\n"
                        f"🚨🚨🚨 DATA LEAKAGE DETECTED! 🚨🚨🚨\n"
                        f"{'='*80}\n"
                        f"Target column '{target_column}' was found in X_train features!\n"
                        f"\nExpected {len(expected_features)} features: {expected_features}\n"
                        f"Got {len(actual_features)} features: {actual_features}\n"
                        f"\nThis means the model can cheat by looking at the answer!\n"
                        f"Training is BLOCKED to prevent invalid results.\n"
                        f"{'='*80}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Check if number of features matches
                if len(actual_features) != len(expected_features):
                    error_msg = (
                        f"\n{'='*80}\n"
                        f"⚠️ FEATURE COUNT MISMATCH!\n"
                        f"{'='*80}\n"
                        f"Expected {len(expected_features)} features, got {len(actual_features)}\n"
                        f"Expected: {expected_features}\n"
                        f"Actual: {actual_features}\n"
                        f"{'='*80}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                logger.info("=" * 80)
                logger.info("✅ POST-SETUP VALIDATION PASSED")
                logger.info(f"   ✓ Target '{target_column}' is NOT in features")
                logger.info(f"   ✓ Feature count matches: {len(actual_features)}")
                logger.info(f"   ✓ No data leakage detected!")
                logger.info("=" * 80)
                
            except Exception as validation_error:
                logger.error(f"❌ POST-SETUP VALIDATION FAILED: {validation_error}")
                raise
            
            logger.info("✅ ML environment setup completed successfully")
            return setup_result
        except Exception as e:
            logger.error(f"❌ Error in ML setup: {e}")
            raise

    async def _compare_models(self, n: int, metric: str, problem_type: str, model_types: List[str] = None) -> Any:
        try:
            if problem_type == 'classification':
                from pycaret.classification import compare_models
                # ✅ CLASSIFICATION MODELS (PyCaret 3.x codes)
                # lr = Logistic Regression ✅
                # rf = Random Forest Classifier ✅
                # et = Extra Trees Classifier ✅
                # gbc = Gradient Boosting Classifier ✅
                # lightgbm = Light Gradient Boosting Machine ✅
                # dt = Decision Tree Classifier ✅
                # nb = Naive Bayes ✅
                # knn = K Neighbors Classifier ✅
                # svm = SVM - Linear Kernel ✅
                # Note: xgboost removed (not installed in environment)
                default_models = ['lr', 'rf', 'et', 'gbc', 'lightgbm', 'dt', 'nb', 'knn', 'svm']
            else:
                from pycaret.regression import compare_models
                # ✅ REGRESSION MODELS (PyCaret 3.x codes)
                # lr = Linear Regression ✅
                # ridge = Ridge Regression ✅
                # lasso = Lasso Regression ✅
                # en = Elastic Net ✅
                # rf = Random Forest Regressor ✅
                # et = Extra Trees Regressor ✅
                # gbr = Gradient Boosting Regressor ✅
                # lightgbm = Light Gradient Boosting Machine ✅
                # dt = Decision Tree Regressor ✅
                # Note: xgboost removed (not installed in environment)
                default_models = ['lr', 'ridge', 'lasso', 'en', 'rf', 'et', 'gbr', 'lightgbm', 'dt']

            # Validate and sanitize model_types
            # Ensure it's a proper list, not None or empty or invalid strings
            if model_types is None or len(model_types) == 0:
                include_models = default_models
                logger.info(f"No model_types specified, using defaults for {problem_type}")
            elif isinstance(model_types, list):
                # Filter out invalid values like None, empty strings, or "Not Available"
                include_models = [
                    m for m in model_types 
                    if m and isinstance(m, str) and m.strip() and m.strip().lower() != 'not available'
                ]
                if not include_models:
                    logger.warning(f"All model_types were invalid, falling back to defaults")
                    include_models = default_models
                else:
                    logger.info(f"Using user-specified model types (validated)")
            else:
                # If model_types is not a list (e.g., a string), use defaults
                logger.warning(f"Invalid model_types type ({type(model_types)}), using defaults")
                include_models = default_models
            
            logger.info(f"COMPARING MODELS:")
            logger.info(f"  - Problem type: {problem_type}")
            logger.info(f"  - Requested models: {n}")
            logger.info(f"  - Model pool: {include_models}")
            logger.info(f"  - Optimization metric: {metric}")
            
            # IMPORTANT: PyCaret may return fewer models if some fail
            # Request more models than necessary to ensure n valid models
            n_select = min(n + 2, len(include_models))  # Request 2 extra as safety margin
            
            logger.info(f"  - n_select (with safety margin): {n_select}")
            
            comparison_result = compare_models(
                include=include_models,
                sort=metric,
                n_select=n_select,  # Request more models
                verbose=False,
                errors='ignore'  # Ignore individual model errors
            )
            
            # Check how many models were returned
            models_returned = len(comparison_result) if isinstance(comparison_result, list) else 1
            logger.info(f"Models returned by PyCaret: {models_returned}")
            
            if models_returned < n:
                logger.warning(
                    f"PyCaret returned {models_returned} models, but {n} were requested. "
                    f"Some models may have failed during training."
                )
            
            return comparison_result
        except Exception as e:
            logger.error(f"Error in model comparison: {e}")
            raise

    async def _create_final_models(self, 
                                  comparison_result: Any,
                                  dataset: pd.DataFrame,
                                  target_column: str, 
                                  n: int,
                                  custom_kwargs: Dict[str, Any],
                                  problem_type: str) -> List[Dict[str, Any]]:
        """Create, finalize, save and register models in MLflow."""
        if problem_type == 'classification':
            from pycaret.classification import finalize_model, pull
        else:
            from pycaret.regression import finalize_model, pull
        
        modelos_finalizados_info = []
        modelos_para_finalizar = comparison_result if isinstance(comparison_result, list) else [comparison_result]
        metrics_df = pull()
        
        logger.info(f"FINALIZING MODELS:")
        logger.info(f"  - Requested: {n} models")
        logger.info(f"  - To finalize: {len(modelos_para_finalizar)} models")
        logger.info(f"  - Model names: {[m.__class__.__name__ for m in modelos_para_finalizar]}")

        for i, modelo in enumerate(modelos_para_finalizar):
            model_class_name = modelo.__class__.__name__
            with mlflow.start_run(run_name=f"Model_{model_class_name}", nested=True) as child_run:
                try:
                    modelo_finalizado = finalize_model(modelo)
                    model_key = self._generate_model_key(modelo_finalizado, dataset, target_column)
                    model_path = self._save_model(modelo_finalizado, model_key)
                    metrics = self._extract_metrics(metrics_df, i, problem_type)
                    
                    # CRITICAL FIX: Calculate ROC AUC manually if missing for multiclass classification
                    if problem_type == 'classification' and (not metrics.get('roc_auc') or metrics.get('roc_auc') == 0.0):
                        try:
                            from sklearn.metrics import roc_auc_score
                            from sklearn.preprocessing import label_binarize
                            
                            # Get predictions on training data for ROC AUC calculation
                            X = dataset.drop(columns=[target_column])
                            y = dataset[target_column]
                            
                            # Get probability predictions
                            y_proba = modelo_finalizado.predict_proba(X)
                            
                            # Binarize labels for multiclass ROC AUC (OvR - One vs Rest)
                            classes = modelo_finalizado.classes_
                            y_bin = label_binarize(y, classes=classes)
                            
                            # Calculate multiclass ROC AUC (macro average)
                            if len(classes) > 2:
                                roc_auc = roc_auc_score(y_bin, y_proba, multi_class='ovr', average='macro')
                            else:
                                roc_auc = roc_auc_score(y, y_proba[:, 1])
                            
                            metrics['roc_auc'] = float(roc_auc)
                            logger.info(f"Calculated ROC AUC manually: {roc_auc:.4f}")
                        except Exception as e:
                            logger.warning(f"Could not calculate ROC AUC manually: {e}")
                    
                    # Generate and save confusion matrix for classification models
                    if problem_type == 'classification':
                        try:
                            confusion_matrix_path = self._generate_confusion_matrix(
                                modelo_finalizado, dataset, target_column, model_key, model_class_name
                            )
                            if confusion_matrix_path:
                                mlflow.log_artifact(str(confusion_matrix_path))
                                logger.info(f"Confusion matrix saved: {confusion_matrix_path}")
                        except Exception as e:
                            logger.warning(f"Could not generate confusion matrix: {e}")
                    
                    mlflow.log_params(custom_kwargs)
                    mlflow.log_metrics(metrics)
                    mlflow.log_artifact(str(model_path))
                    mlflow.set_tag("model_class", model_class_name)
                    
                    mlflow.sklearn.log_model(
                        sk_model=modelo_finalizado,
                        artifact_path="model",
                        registered_model_name=model_class_name
                    )

                    modelo_info = {
                        'key': model_key, 'model': model_class_name,
                        'pickle_path': str(model_path), **metrics
                    }
                    modelos_finalizados_info.append(modelo_info)
                    mlflow.set_tag("status", "SUCCESS")
                    logger.info(f"Model finalized and registered: {model_class_name}")

                except Exception as e:
                    mlflow.set_tag("status", "FAILED")
                    logger.error(f"Error finalizing model {model_class_name}: {e}")
                    continue
        
        logger.info(f"FINALIZATION COMPLETE:")
        logger.info(f"  - Successfully finalized: {len(modelos_finalizados_info)} models")
        logger.info(f"  - Failed: {len(modelos_para_finalizar) - len(modelos_finalizados_info)} models")
        
        if not modelos_finalizados_info:
            raise ValueError("No models were successfully finalized")
        
        if len(modelos_finalizados_info) < n:
            logger.warning(
                f"Only {len(modelos_finalizados_info)} models finalized successfully, "
                f"but {n} were requested. Check logs for errors."
            )
        
        return modelos_finalizados_info

    def _extract_metrics(self, metrics_df: pd.DataFrame, model_index: int, problem_type: str) -> Dict[str, float]:
        """Extract metrics from results DataFrame with 'fail fast' policy."""
        try:
            if model_index >= len(metrics_df): model_index = 0
            row = metrics_df.iloc[model_index]
            
            # Log available columns for debugging
            logger.info(f"Available metric columns: {list(metrics_df.columns)}")
            logger.info(f"Row data for index {model_index}: {dict(row)}")
            
            if problem_type == 'classification':
                # Try multiple column names for ROC AUC (PyCaret may use different names)
                roc_auc = 0.0
                for col_name in ['AUC', 'ROC AUC', 'roc_auc', 'ROCAUC']:
                    if col_name in row.index:
                        roc_auc = float(row.get(col_name, 0.0))
                        if roc_auc > 0:
                            break
                
                metrics = {
                    'accuracy': float(row.get('Accuracy', 0.0)),
                    'precision': float(row.get('Prec.', row.get('Precision', 0.0))),
                    'recall': float(row.get('Recall', 0.0)),
                    'f1_score': float(row.get('F1', row.get('F1 Score', 0.0))),
                    'roc_auc': roc_auc
                }
                
                logger.info(f"Extracted classification metrics: {metrics}")
                return metrics
            else:
                r2 = float(row.get('R2', 0.0))
                return {
                    'accuracy': r2,  # Keep for backward compatibility
                    'r2': r2,
                    'mae': float(row.get('MAE', 0.0)),
                    'mse': float(row.get('MSE', 0.0)),
                    'rmse': float(row.get('RMSE', 0.0))
                }
        except Exception as e:
            logger.error(f"Critical error extracting metrics from PyCaret: {e}", exc_info=True)
            raise ValueError(f"Failed to extract metrics from training result: {e}")

    def _generate_model_key(self, modelo: Any, dataset: pd.DataFrame, target_column: str) -> str:
        model_info = {
            "model_name": modelo.__class__.__name__, "dataset_shape": str(dataset.shape),
            "target_column": target_column, "timestamp": datetime.now().isoformat()
        }
        return json_2_sha256_key(model_info)
    
    def _generate_confusion_matrix(self, model: Any, dataset: pd.DataFrame, target_column: str, 
                                   model_key: str, model_name: str) -> Path:
        """
        Generate and save confusion matrix visualization for classification models.
        
        Args:
            model: Trained classification model
            dataset: Training dataset
            target_column: Name of target column
            model_key: Unique model identifier
            model_name: Model class name
            
        Returns:
            Path to saved confusion matrix image
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            
            # Prepare data
            X = dataset.drop(columns=[target_column])
            y_true = dataset[target_column]
            
            # Get predictions
            y_pred = model.predict(X)
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot confusion matrix with percentages
            cm_display = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=model.classes_
            )
            cm_display.plot(ax=ax, cmap='Blues', values_format='d')
            
            # Add percentages
            for i in range(len(model.classes_)):
                for j in range(len(model.classes_)):
                    total = cm[i, :].sum()
                    percentage = (cm[i, j] / total * 100) if total > 0 else 0
                    ax.text(j, i + 0.3, f'({percentage:.1f}%)', 
                           ha='center', va='center', fontsize=9, color='gray')
            
            # Styling
            ax.set_title(f'Confusion Matrix - {model_name}\n'
                        f'Training Data (n={len(y_true)})', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
            
            # Add summary statistics
            accuracy = (cm.diagonal().sum() / cm.sum()) * 100
            fig.text(0.5, 0.02, f'Overall Accuracy: {accuracy:.2f}%', 
                    ha='center', fontsize=10, style='italic')
            
            plt.tight_layout()
            
            # Save to file
            cm_dir = Path("tmp/confusion_matrices")
            cm_dir.mkdir(parents=True, exist_ok=True)
            cm_path = cm_dir / f"cm_{model_key[:16]}_{model_name}.png"
            
            plt.savefig(cm_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Confusion matrix generated: {cm_path}")
            logger.info(f"  - Accuracy: {accuracy:.2f}%")
            logger.info(f"  - Matrix shape: {cm.shape}")
            
            return cm_path
            
        except Exception as e:
            logger.error(f"Error generating confusion matrix: {e}", exc_info=True)
            return None
        return json_2_sha256_key(model_info)

    def _save_model(self, modelo: Any, model_key: str) -> Path:
        models_dir = Path("tmp/loaded_models")
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{model_key}.pkl"
        joblib.dump(modelo, model_path)
        logger.info(f"Model saved with joblib: {model_path}")
        return model_path

    def _adjust_metric_for_problem_type(self, problem_type, metric_name):
        """Adjust optimization metric based on problem type."""
        if problem_type == 'regression':
            metric_mapping = {'Accuracy': 'R2', 'R2': 'R2', 'MAE': 'MAE', 'RMSE': 'RMSE'}
            return metric_mapping.get(metric_name, 'R2')
        else:
            return metric_name if metric_name in ['Accuracy', 'Precision', 'Recall', 'F1'] else 'Accuracy'

    def _separate_kwargs(self, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Separate kwargs into PyCaret setup kwargs and custom kwargs.
        
        Returns:
            Tuple of (pycaret_kwargs, custom_kwargs)
        """
        pycaret_setup_params = {
            'normalize', 'transformation', 'transform_target', 'ignore_features',
            'remove_outliers', 'outliers_threshold', 'remove_multicollinearity',
            'multicollinearity_threshold', 'pca', 'pca_components', 'feature_selection',
            'feature_selection_method', 'feature_selection_threshold', 'polynomial_features',
            'polynomial_degree', 'low_variance_threshold', 'group_features', 'drop_groups',
            'remove_perfect_collinearity', 'fix_imbalance', 'fix_imbalance_method'
        }
        
        pycaret_kwargs = {k: v for k, v in kwargs.items() if k in pycaret_setup_params}
        custom_kwargs = {k: v for k, v in kwargs.items() if k not in pycaret_setup_params}
        
        return pycaret_kwargs, custom_kwargs

automl_handler = AutoMLHandler()