"""
Streaming System with Graceful Fallback

Implements data streaming with Kafka and simulation mode when Kafka is unavailable.
Allows the application to work in environments with or without Kafka infrastructure.
"""
import json
import asyncio
import logging
import threading
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
    logger.info("Kafka available - full streaming mode activated")
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("Kafka unavailable - simulation mode activated")
    
    class KafkaProducer:
        def __init__(self, **kwargs):
            pass
        def send(self, **kwargs):
            return MockFuture()
        def close(self):
            pass
    
    class KafkaConsumer:
        def __init__(self, *args, **kwargs):
            pass
        def poll(self, **kwargs):
            return {}
        def close(self):
            pass
    
    class KafkaError(Exception):
        pass
    
    class MockFuture:
        def get(self, timeout=None):
            return MockRecordMetadata()
    
    class MockRecordMetadata:
        def __init__(self):
            self.partition = 0
            self.offset = 0

@dataclass
class StreamingEvent:
    """
    Standardized streaming event.
    
    Unified structure for all system events:
    - Model predictions
    - Drift detection
    - Performance metrics
    - Retraining events
    """
    event_type: str
    model_key: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary for transmission."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

class KafkaStreamingHandler:
    """
    Handler for data streaming with graceful fallback.
    
    Features:
    - Event publishing to Kafka topics
    - Event consumption with asynchronous handlers
    - Simulation mode when Kafka is unavailable
    - Connection and consumer thread management
    """
    
    def __init__(self, 
                 bootstrap_servers: str = "localhost:9092",
                 client_id: str = "automlops-api"):
        """
        Initialize streaming handler.
        
        Args:
            bootstrap_servers: Kafka server addresses
            client_id: Unique client identifier
        """
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self.producer: Optional[KafkaProducer] = None
        self.consumers: Dict[str, KafkaConsumer] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.is_running = False
        self.kafka_available = KAFKA_AVAILABLE
        self._consumer_threads: List[threading.Thread] = []
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kafka-consumer")
        
        self._local_events: List[StreamingEvent] = []
        self._max_local_events = 1000
        
        self.topics = {
            'predictions': 'automlops.predictions',
            'model_events': 'automlops.model.events',
            'drift_detection': 'automlops.drift.detection',
            'performance_metrics': 'automlops.performance.metrics',
            'retraining_events': 'automlops.retraining.events'
        }
    
    async def initialize(self) -> bool:
        """
        Initialize Kafka connection or simulation mode.
        
        Fallback strategy:
        1. Attempt Kafka connection
        2. If fail, activate simulation mode
        3. Application continues in both cases
        
        Returns:
            True if initialization was successful
        """
        if not self.kafka_available:
            logger.info("Kafka unavailable - initializing in simulation mode")
            self.is_running = True
            return True
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                client_id=f"{self.client_id}-producer",
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if isinstance(k, str) else k,
                retries=3,
                acks='all',
                compression_type='gzip',
                batch_size=16384,
                linger_ms=10
            )
            
            logger.info(f"Kafka producer initialized: {self.bootstrap_servers}")
            self.is_running = True
            return True
            
        except Exception as e:
            logger.warning(f"Failed to connect to Kafka, using simulation mode: {e}")
            self.kafka_available = False
            self.is_running = True
            return True
    
    async def shutdown(self):
        """Terminate Kafka connections."""
        self.is_running = False
        
        if self.kafka_available and hasattr(self, 'producer') and self.producer:
            for consumer in self.consumers.values():
                try:
                    consumer.close()
                except Exception as e:
                    logger.error(f"Error closing consumer: {e}")
            
            for thread in self._consumer_threads:
                try:
                    thread.join(timeout=5)
                except Exception as e:
                    logger.error(f"Error terminating consumer thread: {e}")
            
            try:
                self.producer.close()
            except Exception as e:
                logger.error(f"Error closing producer: {e}")
        
        try:
            self._thread_pool.shutdown(wait=True, timeout=10)
        except Exception as e:
            logger.error(f"Error terminating thread pool: {e}")
        
        logger.info("Streaming handler terminated")
    
    async def publish_prediction_event(self, 
                                     model_key: str, 
                                     predictions: List[Any],
                                     input_data: List[Dict[str, Any]],
                                     latency_ms: float,
                                     confidence_scores: Optional[List[float]] = None) -> bool:
        """
        Publish prediction event to streaming system.
        
        Captured data:
        - Prediction results
        - Input data used
        - Latency metrics
        - Confidence scores when available
        
        Args:
            model_key: Unique model identifier
            predictions: Prediction results
            input_data: Input data used
            latency_ms: Processing time in milliseconds
            confidence_scores: Prediction confidence scores
            
        Returns:
            True if event was published successfully
        """
        event = StreamingEvent(
            event_type="prediction",
            model_key=model_key,
            timestamp=datetime.utcnow(),
            data={
                "predictions": predictions,
                "input_data": input_data,
                "latency_ms": latency_ms,
                "confidence_scores": confidence_scores,
                "batch_size": len(predictions)
            },
            metadata={
                "api_version": "3.0.0",
                "source": "automlops-api",
                "kafka_mode": "real" if self.kafka_available else "simulation"
            }
        )
        
        return await self._publish_event(self.topics['predictions'], model_key, event)
    
    async def publish_drift_detection(self,
                                    model_key: str,
                                    drift_detected: bool,
                                    drift_score: float,
                                    feature_drifts: Dict[str, Any]) -> bool:
        """
        Publish drift detection event.
        
        Captured data:
        - Drift detection indicator
        - Drift score
        - Drifts per feature
        
        Args:
            model_key: Unique model identifier
            drift_detected: Indicator if drift was detected
            drift_score: Calculated drift score
            feature_drifts: Drift details per feature
            
        Returns:
            True if event was published successfully
        """
        event = StreamingEvent(
            event_type="drift_detection",
            model_key=model_key,
            timestamp=datetime.utcnow(),
            data={
                "drift_detected": drift_detected,
                "drift_score": drift_score,
                "feature_drifts": feature_drifts,
                "threshold_exceeded": drift_score > 0.1
            }
        )
        
        return await self._publish_event(self.topics['drift_detection'], model_key, event)
    
    async def publish_performance_metrics(self,
                                        model_key: str,
                                        metrics: Dict[str, float],
                                        degradation_detected: bool) -> bool:
        """
        Publish model performance metrics.
        
        Args:
            model_key: Unique model identifier
            metrics: Performance metrics
            degradation_detected: If degradation was detected
            
        Returns:
            True if event was published successfully
        """
        event = StreamingEvent(
            event_type="performance_metrics",
            model_key=model_key,
            timestamp=datetime.utcnow(),
            data={
                "metrics": metrics,
                "degradation_detected": degradation_detected,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return await self._publish_event(self.topics['performance_metrics'], model_key, event)
    
    async def publish_retraining_event(self,
                                     model_key: str,
                                     event_type: str,
                                     old_metrics: Optional[Dict[str, float]] = None,
                                     new_metrics: Optional[Dict[str, float]] = None) -> bool:
        """
        Publish retraining event.
        
        Args:
            model_key: Unique model identifier
            event_type: Retraining event type
            old_metrics: Old metrics
            new_metrics: New metrics
            
        Returns:
            True if event was published successfully
        """
        data = {
            "retraining_event_type": event_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if old_metrics:
            data["old_metrics"] = old_metrics
        if new_metrics:
            data["new_metrics"] = new_metrics
            data["improvement"] = self._calculate_improvement(old_metrics or {}, new_metrics)
        
        event = StreamingEvent(
            event_type="retraining_event",
            model_key=model_key,
            timestamp=datetime.utcnow(),
            data=data
        )
        
        return await self._publish_event(self.topics['retraining_events'], model_key, event)
    
    def subscribe_to_topic(self, 
                          topic_key: str, 
                          handler: Callable[[StreamingEvent], None],
                          consumer_group: str = "automlops-consumers") -> bool:
        """
        Subscribe to a topic to consume events.
        
        Args:
            topic_key: Topic key
            handler: Function to process events
            consumer_group: Consumer group
            
        Returns:
            True if subscription was successful
        """
        try:
            if not self.kafka_available:
                logger.info(f"Simulating topic subscription: {topic_key}")
                return True
            
            topic = self.topics.get(topic_key)
            if not topic:
                logger.error(f"Unknown topic: {topic_key}")
                return False
            
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=consumer_group,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest'
            )
            
            self.consumers[topic_key] = consumer
            
            if topic_key not in self.event_handlers:
                self.event_handlers[topic_key] = []
            self.event_handlers[topic_key].append(handler)
            
            consumer_thread = threading.Thread(
                target=self._consume_messages,
                args=(topic_key, consumer),
                daemon=True
            )
            consumer_thread.start()
            self._consumer_threads.append(consumer_thread)
            
            logger.info(f"Subscribed to topic: {topic}")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to topic {topic_key}: {e}")
            return False
    
    def _consume_messages(self, topic_key: str, consumer: KafkaConsumer):
        """Consume messages in separate thread."""
        try:
            for message in consumer:
                try:
                    event_data = message.value
                    
                    for handler in self.event_handlers.get(topic_key, []):
                        try:
                            event = StreamingEvent(**event_data)
                            handler(event)
                        except Exception as e:
                            logger.error(f"Error processing event with handler: {e}")
                            
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except Exception as e:
            logger.error(f"Error in topic consumer {topic_key}: {e}")
    
    async def _publish_event(self, topic: str, key: str, event: StreamingEvent) -> bool:
        """Publish event to topic or store locally."""
        try:
            if self.kafka_available and self.producer:
                future = self.producer.send(
                    topic=topic,
                    key=key,
                    value=event.to_dict()
                )
                
                record_metadata = future.get(timeout=10)
                logger.debug(f"Event published: {topic} - partition: {record_metadata.partition}")
                return True
            else:
                self._local_events.append(event)
                
                if len(self._local_events) > self._max_local_events:
                    self._local_events.pop(0)
                
                logger.debug(f"Event stored locally: {event.event_type}")
                return True
                
        except Exception as e:
            logger.error(f"Error publishing event: {e}")
            return False
    
    def _calculate_improvement(self, old_metrics: Dict[str, float], new_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate improvement between old and new metrics."""
        improvement = {}
        
        for metric_name in set(old_metrics.keys()).intersection(new_metrics.keys()):
            old_value = old_metrics[metric_name]
            new_value = new_metrics[metric_name]
            
            if old_value > 0:
                improvement[metric_name] = (new_value - old_value) / old_value
            else:
                improvement[metric_name] = 0.0
        
        return improvement
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Return connection status."""
        return {
            "kafka_available": self.kafka_available,
            "is_running": self.is_running,
            "mode": "kafka" if self.kafka_available else "simulation",
            "topics": list(self.topics.keys()),
            "active_consumers": len(self.consumers),
            "local_events_count": len(self._local_events),
            "bootstrap_servers": self.bootstrap_servers
        }
    
    def get_local_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return locally stored events (simulation mode)."""
        return [event.to_dict() for event in self._local_events[-limit:]]

kafka_handler = KafkaStreamingHandler()