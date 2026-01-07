"""
Common schemas used throughout the application
"""
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict

class ErrorResponse(BaseModel):
    """Standard error response"""
    erro: bool = Field(True, description="Indicates that an error occurred")
    mensagem: str = Field(..., description="Error message")
    codigo_status: int = Field(..., description="HTTP status code")
    timestamp: str = Field(..., description="Error timestamp")
    caminho: Optional[str] = Field(None, description="Request path")
    detalhes: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class SuccessResponse(BaseModel):
    """Standard success response"""
    sucesso: bool = Field(True, description="Indicates operation success")
    mensagem: str = Field(..., description="Success message")
    timestamp: str = Field(..., description="Operation timestamp")
    dados: Optional[Dict[str, Any]] = Field(None, description="Additional data")