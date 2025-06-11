"""
Extensible Optimization Event System

Simple core with strong extension points for future capabilities.
This module provides basic event logging with clean interfaces for
adding sophisticated analysis capabilities as plugins.
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import logging
from datetime import datetime
from .types import OptimizationEvent, EventHandler

logger = logging.getLogger(__name__)


class EventRegistry:
    """Registry for event handlers with extension points."""
    
    def __init__(self):
        self.handlers: Dict[str, List[EventHandler]] = {}
        self.global_handlers: List[EventHandler] = []
        self._event_count = 0
    
    def register_handler(self, event_type: str, handler: EventHandler) -> None:
        """Register event handler for specific event type."""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
        logger.debug(f"Registered handler for {event_type}")
    
    def register_global_handler(self, handler: EventHandler) -> None:
        """Register handler for all events (extension point)."""
        self.global_handlers.append(handler)
        logger.debug("Registered global event handler")
    
    def emit_event(self, event: OptimizationEvent) -> None:
        """Emit event to all relevant handlers."""
        self._event_count += 1
        
        # Type-specific handlers
        for handler in self.handlers.get(event.event_type, []):
            try:
                if handler.should_handle(event):
                    handler.handle_event(event)
            except Exception as e:
                logger.error(f"Handler error for {event.event_type}: {e}")
        
        # Global handlers (extension point)
        for handler in self.global_handlers:
            try:
                if handler.should_handle(event):
                    handler.handle_event(event)
            except Exception as e:
                logger.error(f"Global handler error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            'total_events': self._event_count,
            'handler_types': list(self.handlers.keys()),
            'global_handlers': len(self.global_handlers),
            'type_handlers': {k: len(v) for k, v in self.handlers.items()}
        }


# Global registry instance
_event_registry = EventRegistry()


# Core event functions
def log_optimization_event(event_type: str, data: Dict[str, Any], 
                          metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Log optimization event with extensible handling.
    
    Args:
        event_type: Type of optimization event
        data: Event data dictionary
        metadata: Optional metadata for extensions
    """
    event = OptimizationEvent(
        timestamp=datetime.now(),
        event_type=event_type,
        data=data,
        metadata=metadata or {}
    )
    _event_registry.emit_event(event)


def log_parameter_change(parameter: str, old_value: Any, new_value: Any,
                        context: Optional[Dict[str, Any]] = None) -> None:
    """Log parameter change (common optimization event)."""
    log_optimization_event('parameter_change', {
        'parameter': parameter,
        'old_value': old_value,
        'new_value': new_value,
        'change_magnitude': abs(float(new_value) - float(old_value)) if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)) else None
    }, metadata=context)


def log_performance_metric(metric_name: str, value: float, 
                         context: Optional[Dict[str, Any]] = None) -> None:
    """Log performance measurement (common optimization event)."""
    log_optimization_event('performance_metric', {
        'metric': metric_name,
        'value': value,
        'context': context or {}
    })


def log_strategy_decision(strategy: str, rationale: str = "",
                         alternatives: Optional[List[str]] = None) -> None:
    """Log strategy selection (common optimization event)."""
    log_optimization_event('strategy_decision', {
        'strategy': strategy,
        'rationale': rationale,
        'alternatives': alternatives or []
    })


def log_dse_event(stage: str, data: Dict[str, Any]) -> None:
    """Log design space exploration events."""
    log_optimization_event('dse_event', {
        'stage': stage,
        **data
    })


# Extension interface functions
def register_event_handler(event_type: str, handler: EventHandler) -> None:
    """Register custom event handler (extension point)."""
    _event_registry.register_handler(event_type, handler)


def register_global_handler(handler: EventHandler) -> None:
    """Register global event handler (extension point)."""
    _event_registry.register_global_handler(handler)


def create_custom_event_type(event_type: str, validator: Optional[callable] = None) -> None:
    """Create custom event type (extension point)."""
    # Future: Add validation and schema support
    logger.info(f"Registered custom event type: {event_type}")
    if validator:
        logger.debug(f"Custom validator registered for {event_type}")


# Simple built-in handlers
class ConsoleHandler(EventHandler):
    """Simple console logging handler."""
    
    def handle_event(self, event: OptimizationEvent) -> None:
        logger.info(f"Event: {event.event_type} - {event.data}")


class MemoryHandler(EventHandler):
    """Simple in-memory event storage."""
    
    def __init__(self, max_events: int = 1000):
        self.events: List[OptimizationEvent] = []
        self.max_events = max_events
    
    def handle_event(self, event: OptimizationEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.max_events:
            self.events.pop(0)
    
    def get_recent_events(self, count: int = 10) -> List[OptimizationEvent]:
        """Get recent events."""
        return self.events[-count:] if self.events else []
    
    def get_events_by_type(self, event_type: str) -> List[OptimizationEvent]:
        """Get events by type."""
        return [e for e in self.events if e.event_type == event_type]
    
    def clear_events(self) -> None:
        """Clear stored events."""
        self.events.clear()


# Default handlers
_memory_handler = MemoryHandler()
_event_registry.register_global_handler(_memory_handler)


def get_recent_events(count: int = 10) -> List[OptimizationEvent]:
    """Get recent events from default memory handler."""
    return _memory_handler.get_recent_events(count)


def get_events_by_type(event_type: str) -> List[OptimizationEvent]:
    """Get events by type from default memory handler."""
    return _memory_handler.get_events_by_type(event_type)


def get_event_stats() -> Dict[str, Any]:
    """Get event system statistics."""
    return _event_registry.get_stats()


def clear_event_history() -> None:
    """Clear event history (useful for testing)."""
    _memory_handler.clear_events()