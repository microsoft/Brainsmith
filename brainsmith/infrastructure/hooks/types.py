"""
Essential Types for Extensible Optimization Hooks

Simple types with clear extension points for future capabilities.
This module provides the foundational types for the hooks system while
maintaining clean interfaces for sophisticated extensions.
"""

from typing import Dict, Any, List, Optional, Protocol
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod


@dataclass
class OptimizationEvent:
    """
    Core optimization event with extension support.
    
    Future extensions can add additional fields or processing
    without breaking the core interface.
    """
    timestamp: datetime
    event_type: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (extension point for serialization)."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'data': self.data,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationEvent':
        """Create from dictionary (extension point for deserialization)."""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            event_type=data['event_type'],
            data=data['data'],
            metadata=data.get('metadata', {})
        )
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """Get value from data or metadata."""
        return self.data.get(key, self.metadata.get(key, default))
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata (extension point)."""
        self.metadata[key] = value


class EventHandler(ABC):
    """
    Abstract event handler interface (strong extension point).
    
    Future capabilities can implement this interface:
    - StatisticalAnalysisHandler
    - MLStrategyHandler  
    - DatabasePersistenceHandler
    - MetricsAggregationHandler
    """
    
    @abstractmethod
    def handle_event(self, event: OptimizationEvent) -> None:
        """Handle optimization event."""
        pass
    
    def should_handle(self, event: OptimizationEvent) -> bool:
        """Filter events (extension point)."""
        return True
    
    def initialize(self) -> None:
        """Initialize handler (extension point)."""
        pass
    
    def cleanup(self) -> None:
        """Cleanup handler (extension point)."""
        pass
    
    def get_name(self) -> str:
        """Get handler name for identification."""
        return self.__class__.__name__


@dataclass
class SimpleMetric:
    """
    Basic performance metric (extensible for future metric types).
    """
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'unit': self.unit
        }
    
    @classmethod
    def from_event(cls, event: OptimizationEvent) -> Optional['SimpleMetric']:
        """Create metric from performance event."""
        if event.event_type == 'performance_metric':
            return cls(
                name=event.data.get('metric', 'unknown'),
                value=event.data.get('value', 0.0),
                timestamp=event.timestamp,
                tags=event.data.get('context', {}),
                unit=event.metadata.get('unit')
            )
        return None


@dataclass
class ParameterChange:
    """Simple parameter change record."""
    parameter: str
    old_value: Any
    new_value: Any
    timestamp: datetime
    change_magnitude: Optional[float] = None
    
    @classmethod
    def from_event(cls, event: OptimizationEvent) -> Optional['ParameterChange']:
        """Create parameter change from event."""
        if event.event_type == 'parameter_change':
            return cls(
                parameter=event.data.get('parameter', 'unknown'),
                old_value=event.data.get('old_value'),
                new_value=event.data.get('new_value'),
                timestamp=event.timestamp,
                change_magnitude=event.data.get('change_magnitude')
            )
        return None


# Extension point: Custom event types
EventTypeRegistry = Dict[str, Dict[str, Any]]  # Future: Event schemas


# Extension point: Plugin interface  
class HooksPlugin(Protocol):
    """Protocol for hooks plugins (future extension point)."""
    
    def install(self) -> None:
        """Install plugin."""
        ...
    
    def uninstall(self) -> None:
        """Uninstall plugin."""
        ...
    
    def get_handlers(self) -> List[EventHandler]:
        """Get plugin event handlers."""
        ...
    
    def get_name(self) -> str:
        """Get plugin name."""
        ...


# Extension point: Event filters and processors
class EventFilter(ABC):
    """Abstract event filter for sophisticated event processing."""
    
    @abstractmethod
    def should_process(self, event: OptimizationEvent) -> bool:
        """Determine if event should be processed."""
        pass


class EventProcessor(ABC):
    """Abstract event processor for complex event analysis."""
    
    @abstractmethod
    def process_event(self, event: OptimizationEvent) -> Optional[Dict[str, Any]]:
        """Process event and return analysis results."""
        pass


# Common event types (extensible)
class EventTypes:
    """Common event type constants (extensible)."""
    
    PARAMETER_CHANGE = "parameter_change"
    PERFORMANCE_METRIC = "performance_metric"
    STRATEGY_DECISION = "strategy_decision"
    DSE_EVENT = "dse_event"
    OPTIMIZATION_START = "optimization_start"
    OPTIMIZATION_END = "optimization_end"
    ERROR_EVENT = "error_event"
    
    # Extension point: Custom event types can be added here
    @classmethod
    def register_custom_type(cls, type_name: str) -> None:
        """Register custom event type."""
        setattr(cls, type_name.upper(), type_name)


# Helper functions for creating common events
def create_parameter_event(parameter: str, old_value: Any, new_value: Any) -> OptimizationEvent:
    """Create parameter change event."""
    return OptimizationEvent(
        timestamp=datetime.now(),
        event_type=EventTypes.PARAMETER_CHANGE,
        data={
            'parameter': parameter,
            'old_value': old_value,
            'new_value': new_value
        }
    )


def create_metric_event(metric_name: str, value: float, context: Optional[Dict] = None) -> OptimizationEvent:
    """Create performance metric event."""
    return OptimizationEvent(
        timestamp=datetime.now(),
        event_type=EventTypes.PERFORMANCE_METRIC,
        data={
            'metric': metric_name,
            'value': value,
            'context': context or {}
        }
    )


def create_strategy_event(strategy: str, rationale: str = "") -> OptimizationEvent:
    """Create strategy decision event."""
    return OptimizationEvent(
        timestamp=datetime.now(),
        event_type=EventTypes.STRATEGY_DECISION,
        data={
            'strategy': strategy,
            'rationale': rationale
        }
    )