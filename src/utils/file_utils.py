"""
File Manipulation Utilities
"""
import uuid
import hashlib
from pathlib import Path

def gerar_nome_arquivo_seguro(nome_arquivo: str, content: bytes = None) -> str:
    """
    Generate unique and safe filename based on content to avoid duplication.
    
    Args:
        nome_arquivo: Original filename
        content: File content (optional)
        
    Returns:
        Sanitized and unique filename
    """
    if not nome_arquivo:
        nome_arquivo = "dataset.csv"
    
    objeto_path = Path(nome_arquivo)
    extensao = objeto_path.suffix or '.csv'
    nome_base = objeto_path.stem or 'dataset'
    
    nome_seguro = "".join([
        c for c in nome_base 
        if c.isalpha() or c.isdigit() or c in ('_', '-')
    ]).rstrip()
    
    if not nome_seguro:
        nome_seguro = "dataset"
    
    if content:
        content_hash = hashlib.md5(content).hexdigest()[:8]
        return f"{nome_seguro}_{content_hash}{extensao}"
    else:
        return f"{nome_seguro}_{uuid.uuid4().hex[:8]}{extensao}"

def verificar_arquivo_existente(nome_arquivo: str, content: bytes, pasta_destino: Path) -> Path:
    """
    Check if file with same content already exists.
    
    Args:
        nome_arquivo: Original filename
        content: File content
        pasta_destino: Destination folder where file will be saved
        
    Returns:
        Path to existing or new file
    """
    content_hash = hashlib.md5(content).hexdigest()
    
    for arquivo_existente in pasta_destino.glob("*.csv"):
        try:
            with open(arquivo_existente, 'rb') as f:
                existing_content = f.read()
                if hashlib.md5(existing_content).hexdigest() == content_hash:
                    return arquivo_existente
        except Exception:
            continue
    
    nome_seguro = gerar_nome_arquivo_seguro(nome_arquivo, content)
    return pasta_destino / nome_seguro