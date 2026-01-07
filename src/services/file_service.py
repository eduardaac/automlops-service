"""
File Management Service
"""
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from sqlalchemy.orm import Session

from src.repositories.model_repository import FileRepository
from src.database.models.File import File as DBFile
from src.utils.file_utils import verificar_arquivo_existente

logger = logging.getLogger(__name__)

class FileService:
    """Service for file management"""
    
    def __init__(self):
        self.file_repo = FileRepository()
    
    def save_uploaded_file(
        self,
        content: bytes,
        filename: str,
        content_type: str,
        target_column: str,
        safe_filename: str,
        file_path: Path,
        db: Session,
        optimization_metric: str = "Accuracy"
    ) -> DBFile:
        """
        Save uploaded file and create database record, avoiding duplicates.
        """
        destination_folder = Path("tmp/files")
        destination_folder.mkdir(parents=True, exist_ok=True)
        
        final_path = verificar_arquivo_existente(filename, content, destination_folder)
        
        if final_path.exists():
            logger.info(f"File with identical content found: {final_path}")
            
            existing_record = db.query(DBFile).filter(
                DBFile.content_path == str(final_path)
            ).first()
            
            if existing_record and existing_record.target_column == target_column:
                logger.info("Compatible target column, reusing existing file")
                
                # Atualizar campos se estiverem vazios (registros antigos)
                if not hasattr(existing_record, 'rows') or existing_record.rows is None or existing_record.rows == 0:
                    try:
                        dataset_info = self.validate_csv_file(final_path, target_column)
                        existing_record.rows = dataset_info['rows']
                        existing_record.columns = len(dataset_info['columns'])
                        existing_record.columns_names = dataset_info['columns']
                        existing_record.shape = dataset_info['shape']
                        db.commit()
                        logger.info(f"Updated existing record with dataset info: {dataset_info['rows']} rows")
                    except Exception as e:
                        logger.warning(f"Could not update existing record: {e}")
                
                return existing_record
        
        if not final_path.exists():
            with open(final_path, 'wb') as f:
                f.write(content)
            logger.info(f"File saved successfully: {final_path}")
        
        try:
            dataset_info = self.validate_csv_file(final_path, target_column)
        except Exception as e:
            if not existing_record and final_path.exists():
                final_path.unlink()
            raise ValueError(f"Error validating file: {str(e)}")
        
        file_data = {
            'name': filename,
            'content_path': str(final_path),
            'size': len(content),
            'content_type': content_type,
            'target_column': target_column,
            'optimization_metric': optimization_metric
        }
        
        db_file = self.file_repo.create(db, file_data)
        
        db_file.rows = dataset_info['rows']
        db_file.columns = len(dataset_info['columns'])
        db_file.columns_names = dataset_info['columns']
        db_file.shape = dataset_info['shape']
        
        return db_file
    
    def validate_csv_file(self, file_path: Path, target_column: str) -> Dict[str, Any]:
        """
        Validate CSV file and return information.
        
        Args:
            file_path: File path
            target_column: Expected target column
            
        Returns:
            Dictionary with dataset information
            
        Raises:
            ValueError: If validation fails
        """
        try:
            if not file_path.exists():
                raise ValueError(f"File not found: {file_path}")
            
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            dataset = None
            
            for encoding in encodings:
                try:
                    dataset = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if dataset is None:
                raise ValueError("Could not read CSV file. Check encoding.")
            
            if dataset.empty:
                raise ValueError("Dataset is empty")
            
            if target_column not in dataset.columns:
                raise ValueError(
                    f"Target column '{target_column}' not found. "
                    f"Available columns: {list(dataset.columns)}"
                )
            
            if len(dataset) < 10:
                raise ValueError(f"Dataset too small: {len(dataset)} rows. Recommended minimum: 100 rows")
            
            if len(dataset.columns) < 2:
                raise ValueError("Dataset must have at least 2 columns (1 feature + 1 target)")
            
            return {
                'shape': dataset.shape,
                'columns': list(dataset.columns),
                'target_column': target_column,
                'rows': len(dataset),
                'memory_usage': dataset.memory_usage(deep=True).sum(),
                'dtypes': dataset.dtypes.to_dict(),
                'null_counts': dataset.isnull().sum().to_dict()
            }
            
        except pd.errors.EmptyDataError:
            raise ValueError("Dataset is empty")
        except pd.errors.ParserError as e:
            raise ValueError(f"Invalid CSV format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error validating file: {str(e)}")
    
    def get_file_by_id(self, file_id: int, db: Session) -> DBFile:
        """
        Get file by ID and verify it exists physically.
        
        Args:
            file_id: File ID
            db: Database session
            
        Returns:
            Found file
            
        Raises:
            ValueError: If file not found or physical file doesn't exist
        """
        file = self.file_repo.get_by_id(db, file_id)
        
        if not file:
            raise ValueError(f"File not found in database: {file_id}")
        
        file_path = Path(file.content_path)
        if not file_path.exists():
            logger.error(f"Physical file not found: {file.content_path}")
            raise ValueError(
                f"Physical file not found: {file.content_path}. "
                f"Database record may be inconsistent."
            )
        
        logger.info(f"File found: {file.name} -> {file.content_path}")
        return file
    
    def get_file_info(self, file_id: int, db: Session) -> Dict[str, Any]:
        """
        Get detailed file information including dataset data.
        """
        file = self.get_file_by_id(file_id, db)
        
        try:
            dataset_info = self.validate_csv_file(Path(file.content_path), file.target_column)
            
            return {
                'file_id': file.id,
                'filename': file.name,
                'file_path': file.content_path,
                'file_size': file.size,
                'content_type': file.content_type,
                'target_column': file.target_column,
                'created_at': file.created_at,
                **dataset_info
            }
            
        except Exception as e:
            logger.error(f"Error getting file {file_id} information: {e}")
            return {
                'file_id': file.id,
                'filename': file.name,
                'file_path': file.content_path,
                'file_size': file.size,
                'content_type': file.content_type,
                'target_column': file.target_column,
                'created_at': file.created_at,
                'error': str(e)
            }