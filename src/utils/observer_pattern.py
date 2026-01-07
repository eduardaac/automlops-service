"""
Observer Pattern Implementation for Proactive Monitoring System

This module implements the fundamental Observer pattern interfaces
to decouple event detection logic from monitoring logic.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Any, Dict

logger = logging.getLogger(__name__)


class Observer(ABC):
    """
    Abstract interface for observers (subscribers).
    
    Observers implement this interface to receive
    notifications of specific system events.
    """
    
    @abstractmethod
    async def update(self, event_data: Dict[str, Any], subject: 'Subject') -> None:
        """
        Method called when an event is triggered by subject.
        
        Args:
            event_data: Event data
            subject: Subject that triggered the event
        """
        pass


class Subject(ABC):
    """
    Abstract interface for event publishers (subjects).
    
    Subjects maintain a list of observers and notify them
    when relevant events occur in the system.
    """
    
    def __init__(self):
        self._observers: List[Observer] = []
    
    def register(self, observer: Observer) -> None:
        """Register an observer."""
        if observer not in self._observers:
            self._observers.append(observer)
            logger.info(f"Observer {observer.__class__.__name__} registered")
    
    def unregister(self, observer: Observer) -> None:
        """Remove an observer."""
        if observer in self._observers:
            self._observers.remove(observer)
            logger.info(f"Observer {observer.__class__.__name__} removed")
    
    async def notify(self, event_data: Dict[str, Any]) -> None:
        """Notify all observers about an event."""
        logger.info(f"Notifying {len(self._observers)} observers about event: {event_data.get('event_type', 'unknown')}")
        
        for observer in self._observers:
            try:
                await observer.update(event_data, self)
            except Exception as e:
                logger.error(f"Error notifying observer {observer.__class__.__name__}: {e}")


class EventType:
    """Constants for system event types."""
    MODEL_DEGRADATION_DETECTED = "model_degradation_detected"
    DATA_DRIFT_DETECTED = "data_drift_detected"
    HIGH_PREDICTION_VOLUME = "high_prediction_volume"
    LOW_CONFIDENCE_PREDICTIONS = "low_confidence_predictions"
    SYSTEM_ERROR = "system_error"