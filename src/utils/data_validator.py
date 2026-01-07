"""
Data Validation Module - Flexible and Dynamic Validator
Validates basic DataFrame integrity without enforcing specific schemas
"""
import logging
import pandas as pd
from typing import List, Optional

logger = logging.getLogger(__name__)

class DataValidator:
    """Flexible DataFrame validator that adapts to any dataset structure."""

    def __init__(self):
        """Initialize validator without rigid schema requirements."""
        self.validation_errors: List[str] = []
        self.warnings: List[str] = []

    def validate_dataframe(self, df: pd.DataFrame, 
                          min_rows: int = 10,
                          min_columns: int = 2,
                          max_null_percentage: float = 0.5,
                          check_nulls: bool = False) -> bool:
        """
        Execute flexible validations on DataFrame.
        
        Args:
            df: DataFrame to validate
            min_rows: Minimum number of rows required (default: 10)
            min_columns: Minimum number of columns required (default: 2)
            max_null_percentage: Maximum percentage of nulls allowed per column (default: 50%)
            check_nulls: Whether to check for null values (default: False for flexibility)
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        self.validation_errors = []
        self.warnings = []
        logger.info(f"Starting flexible DataFrame validation... Shape: {df.shape}")

        # 1. Check if DataFrame is empty
        if df.empty:
            self.validation_errors.append("DataFrame is empty (0 rows)")
            logger.error("Validation failed: Empty DataFrame")
            return False

        # 2. Check minimum row count
        if len(df) < min_rows:
            self.validation_errors.append(
                f"Insufficient data: {len(df)} rows (minimum: {min_rows})"
            )

        # 3. Check minimum column count
        if len(df.columns) < min_columns:
            self.validation_errors.append(
                f"Insufficient features: {len(df.columns)} columns (minimum: {min_columns})"
            )

        # 4. Check for duplicate column names
        if df.columns.duplicated().any():
            duplicate_cols = df.columns[df.columns.duplicated()].tolist()
            self.validation_errors.append(f"Duplicate column names found: {duplicate_cols}")

        # 5. Optional: Check for excessive null values
        if check_nulls:
            for col in df.columns:
                null_pct = df[col].isnull().sum() / len(df)
                if null_pct > max_null_percentage:
                    self.warnings.append(
                        f"Column '{col}' has {null_pct:.1%} null values (threshold: {max_null_percentage:.1%})"
                    )

        # 6. Check for columns with all same values (zero variance)
        for col in df.select_dtypes(include=['number']).columns:
            if df[col].nunique() == 1:
                self.warnings.append(f"Column '{col}' has zero variance (all values are identical)")

        # Log results
        if self.validation_errors:
            logger.error("Data validation failed!")
            for error in self.validation_errors:
                logger.error(f"  - {error}")
            return False
        
        if self.warnings:
            logger.warning("Data validation passed with warnings:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        logger.info(f"Data validation completed successfully. Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        return True

    def infer_schema(self, df: pd.DataFrame) -> dict:
        """
        Infer schema from DataFrame dynamically.
        
        Returns:
            dict: Dictionary with column names as keys and inferred types as values
        """
        schema = {}
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                schema[col] = 'integer'
            elif pd.api.types.is_float_dtype(dtype):
                schema[col] = 'float'
            elif pd.api.types.is_bool_dtype(dtype):
                schema[col] = 'boolean'
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                schema[col] = 'datetime'
            else:
                schema[col] = 'string'
        
        logger.info(f"Inferred schema: {schema}")
        return schema

# Singleton instance with flexible validation (no hardcoded schema)
data_validator = DataValidator()