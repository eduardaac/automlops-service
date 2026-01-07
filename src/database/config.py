# src/database/config.py
from sqlalchemy import create_engine, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import registry
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://automlops_user:1598@127.0.0.1:5432/automlops_db"
)

logger.info(f"Using database: {SQLALCHEMY_DATABASE_URL.split('@')[1] if '@' in SQLALCHEMY_DATABASE_URL else 'local'}")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

mapper_registry = registry()

def init_db():
    """Initialize database and create tables."""
    try:
        from src.database.models.File import File
        from src.database.models.Result import Result
        from src.database.models.Alert import Alert
        from src.database.models.PerformanceLog import PerformanceLog
        
        mapper_registry.configure()
        
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        logger.info(f"Existing tables: {existing_tables}")
        
        if 'results' in existing_tables:
            result_columns = [col['name'] for col in inspector.get_columns('results')]
            logger.info(f"Results columns: {result_columns}")
            
            required_columns = ['precision', 'recall', 'f1_score', 'roc_auc', 'cross_val_roc_auc']
            missing_columns = [col for col in required_columns if col not in result_columns]
            
            if missing_columns:
                logger.info(f"Missing columns detected: {missing_columns}. Recreating results table...")
                
                from sqlalchemy import text
                with engine.connect() as conn:
                    conn.execute(text("DROP TABLE IF EXISTS results"))
                    conn.commit()
        
        Base.metadata.create_all(bind=engine)
       
        inspector = inspect(engine)
        
        if 'files' in inspector.get_table_names():
            files_columns = [col['name'] for col in inspector.get_columns('files')]
            logger.info(f"Files columns: {files_columns}")
            
            # Check for missing columns in files table
            if 'optimization_metric' not in files_columns:
                logger.warning("optimization_metric column missing in files table. Adding it...")
                from sqlalchemy import text
                try:
                    with engine.connect() as conn:
                        conn.execute(text("ALTER TABLE files ADD COLUMN optimization_metric VARCHAR DEFAULT 'Accuracy'"))
                        conn.commit()
                    logger.info("optimization_metric column added successfully")
                except Exception as e:
                    logger.error(f"Failed to add optimization_metric column: {e}")
                    raise
            
            if 'created_at' not in files_columns:
                logger.error("created_at column missing in files table")
                raise ValueError("Failed to create files table with created_at")
        
        if 'results' in inspector.get_table_names():
            result_columns = [col['name'] for col in inspector.get_columns('results')]
            logger.info(f"Results columns: {result_columns}")
        
        if 'alerts' in inspector.get_table_names():
            alerts_columns = [col['name'] for col in inspector.get_columns('alerts')]
            logger.info(f"Alerts columns: {alerts_columns}")
            
            # Check for missing columns in alerts table
            if 'retraining_job_id' not in alerts_columns:
                logger.warning("retraining_job_id column missing in alerts table. Adding it...")
                from sqlalchemy import text
                try:
                    with engine.connect() as conn:
                        conn.execute(text("ALTER TABLE alerts ADD COLUMN retraining_job_id INTEGER REFERENCES files(id)"))
                        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_alerts_retraining_job_id ON alerts(retraining_job_id)"))
                        conn.commit()
                    logger.info("retraining_job_id column added successfully")
                except Exception as e:
                    logger.error(f"Failed to add retraining_job_id column: {e}")
                    raise
        
        if 'performance_logs' in inspector.get_table_names():
            perf_columns = [col['name'] for col in inspector.get_columns('performance_logs')]
            logger.info(f"Performance logs columns: {perf_columns}")
        
        logger.info("Database initialized successfully.")
        
    except Exception as e:
        logger.error(f"Database init error: {str(e)}")
        raise

def get_db():
    """Provide database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()