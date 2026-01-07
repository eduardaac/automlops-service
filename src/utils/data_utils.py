"""
Data Utilities - Centralized data manipulation functions
Provides reusable, flexible utilities for data processing
"""
import pandas as pd
from typing import List, Union


# Target columns that should be excluded from features
# CO2_daily_kg is the REGRESSION TARGET (what models predict)
# CO2_category is the classification target (for classification models)
DEFAULT_TARGET_COLUMNS = ["Target", "target", "label", "Label", "CLASS", "class", "y", "Y", "CO2_daily_kg", "CO2_category"]


def remove_target_columns(
    data: Union[pd.DataFrame, List[str]], 
    target_columns: List[str] = None
) -> Union[pd.DataFrame, List[str]]:
    """
    Remove target columns from DataFrame or column list.
    
    Sistema flexível que:
    - Aceita DataFrame ou lista de colunas
    - Permite customização das colunas target via parâmetro
    - Remove apenas colunas que existem (evita erros)
    - Mantém dados originais intactos (retorna cópia)
    
    Args:
        data: DataFrame or list of column names
        target_columns: List of target column names to remove. 
                       If None, uses DEFAULT_TARGET_COLUMNS.
    
    Returns:
        DataFrame or list without target columns
        
    Examples:
        >>> df = pd.DataFrame({'a': [1,2], 'target': [0,1]})
        >>> clean_df = remove_target_columns(df)
        
        >>> cols = ['feature1', 'feature2', 'label']
        >>> clean_cols = remove_target_columns(cols)
    """
    if target_columns is None:
        target_columns = DEFAULT_TARGET_COLUMNS
    
    # Caso 1: DataFrame
    if isinstance(data, pd.DataFrame):
        # Remove apenas colunas que existem no DataFrame
        cols_to_drop = [col for col in target_columns if col in data.columns]
        if cols_to_drop:
            return data.drop(columns=cols_to_drop)
        return data
    
    # Caso 2: Lista de colunas
    elif isinstance(data, list):
        return [col for col in data if col not in target_columns]
    
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected DataFrame or list.")


def get_feature_columns(
    df: pd.DataFrame, 
    target_columns: List[str] = None
) -> List[str]:
    """
    Get only feature column names (exclude targets).
    
    Args:
        df: Input DataFrame
        target_columns: List of target column names. If None, uses DEFAULT_TARGET_COLUMNS.
    
    Returns:
        List of feature column names
    """
    if target_columns is None:
        target_columns = DEFAULT_TARGET_COLUMNS
    
    return [col for col in df.columns if col not in target_columns]


def has_target_column(df: pd.DataFrame, target_columns: List[str] = None) -> bool:
    """
    Check if DataFrame contains any target column.
    
    Args:
        df: Input DataFrame
        target_columns: List of target column names. If None, uses DEFAULT_TARGET_COLUMNS.
    
    Returns:
        True if at least one target column exists
    """
    if target_columns is None:
        target_columns = DEFAULT_TARGET_COLUMNS
    
    return any(col in df.columns for col in target_columns)
