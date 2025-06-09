"""
Core Metrics Framework
Foundation classes and interfaces for comprehensive metrics collection and management.
"""

import os
import sys
import json
import time
import logging
import threading
import sqlite3
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field, asdict
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
import uuid
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    QUALITY = "quality"
    TIMING = "timing"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    POWER = "power"
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    UTILIZATION = "utilization"


class MetricScope(Enum):
    """Scope levels for metrics."""
    GLOBAL = "global"
    BUILD = "build"
    TRANSFORMATION = "transformation"
    KERNEL = "kernel"
    OPERATOR = "operator"
    PLATFORM = "platform"


@dataclass
class MetricValue:
    """Represents a single metric value with metadata."""
    name: str
    value: Union[int, float, str, bool]
    unit: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    scope: MetricScope = MetricScope.GLOBAL
    metric_type: MetricType = MetricType.PERFORMANCE
    confidence: Optional[float] = None
    source: Optional[str] = None


@dataclass
class MetricCollection:
    """Collection of related metrics."""
    collection_id: str
    name: str
    description: str
    metrics: List[MetricValue] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_metric(self, metric: MetricValue):
        """Add metric to collection."""
        self.metrics.append(metric)
    
    def get_metrics_by_type(self, metric_type: MetricType) -> List[MetricValue]:
        """Get metrics of specific type."""
        return [m for m in self.metrics if m.metric_type == metric_type]
    
    def get_metrics_by_scope(self, scope: MetricScope) -> List[MetricValue]:
        """Get metrics of specific scope."""
        return [m for m in self.metrics if m.scope == scope]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'collection_id': self.collection_id,
            'name': self.name,
            'description': self.description,
            'metrics': [asdict(m) for m in self.metrics],
            'created_at': self.created_at,
            'tags': self.tags,
            'metadata': self.metadata
        }


class MetricsCollector(ABC):
    """Abstract base class for metrics collectors."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.enabled = True
        self.collection_callbacks = []
    
    @abstractmethod
    def collect_metrics(self, context: Dict[str, Any]) -> MetricCollection:
        """Collect metrics for given context."""
        pass
    
    @abstractmethod
    def get_supported_metrics(self) -> List[str]:
        """Get list of metrics this collector can provide."""
        pass
    
    def add_collection_callback(self, callback: Callable[[MetricCollection], None]):
        """Add callback for when metrics are collected."""
        self.collection_callbacks.append(callback)
    
    def _notify_collection_callbacks(self, collection: MetricCollection):
        """Notify all collection callbacks."""
        for callback in self.collection_callbacks:
            try:
                callback(collection)
            except Exception as e:
                logger.warning(f"Metrics collection callback failed: {e}")
    
    def set_enabled(self, enabled: bool):
        """Enable or disable this collector."""
        self.enabled = enabled
    
    def is_enabled(self) -> bool:
        """Check if collector is enabled."""
        return self.enabled


class MetricsRegistry:
    """Registry for managing metrics collectors."""
    
    def __init__(self):
        self.collectors = {}
        self.collector_instances = {}
        self.collection_history = []
        self.lock = threading.Lock()
    
    def register_collector(self, collector_class: Type[MetricsCollector], name: Optional[str] = None):
        """Register a metrics collector class."""
        collector_name = name or collector_class.__name__
        with self.lock:
            self.collectors[collector_name] = collector_class
            logger.info(f"Registered metrics collector: {collector_name}")
    
    def create_collector(self, collector_name: str, config: Optional[Dict[str, Any]] = None) -> MetricsCollector:
        """Create instance of registered collector."""
        with self.lock:
            if collector_name not in self.collectors:
                raise ValueError(f"Unknown collector: {collector_name}")
            
            collector_class = self.collectors[collector_name]
            instance = collector_class(collector_name, config)
            self.collector_instances[collector_name] = instance
            
            return instance
    
    def get_collector(self, collector_name: str) -> Optional[MetricsCollector]:
        """Get existing collector instance."""
        return self.collector_instances.get(collector_name)
    
    def list_collectors(self) -> List[str]:
        """List all registered collector names."""
        return list(self.collectors.keys())
    
    def list_active_collectors(self) -> List[str]:
        """List active collector instances."""
        return [name for name, collector in self.collector_instances.items() if collector.is_enabled()]
    
    def collect_all_metrics(self, context: Dict[str, Any]) -> List[MetricCollection]:
        """Collect metrics from all active collectors."""
        collections = []
        
        for name, collector in self.collector_instances.items():
            if collector.is_enabled():
                try:
                    collection = collector.collect_metrics(context)
                    collections.append(collection)
                    self.collection_history.append(collection)
                except Exception as e:
                    logger.error(f"Failed to collect metrics from {name}: {e}")
        
        # Trim history
        if len(self.collection_history) > 10000:
            self.collection_history = self.collection_history[-5000:]
        
        return collections


class MetricsAggregator:
    """Aggregate metrics from multiple sources."""
    
    def __init__(self):
        self.aggregation_functions = {
            'sum': self._sum_aggregation,
            'avg': self._avg_aggregation,
            'min': self._min_aggregation,
            'max': self._max_aggregation,
            'count': self._count_aggregation,
            'std': self._std_aggregation
        }
    
    def aggregate_collections(self, 
                             collections: List[MetricCollection],
                             group_by: Optional[str] = None,
                             aggregation: str = 'avg') -> MetricCollection:
        """Aggregate multiple metric collections."""
        
        if not collections:
            return MetricCollection(
                collection_id=str(uuid.uuid4()),
                name="empty_aggregation",
                description="Empty aggregation result"
            )
        
        # Flatten all metrics
        all_metrics = []
        for collection in collections:
            all_metrics.extend(collection.metrics)
        
        # Group metrics
        if group_by:
            grouped_metrics = self._group_metrics(all_metrics, group_by)
        else:
            grouped_metrics = {'all': all_metrics}
        
        # Apply aggregation
        aggregated_metrics = []
        agg_func = self.aggregation_functions.get(aggregation, self._avg_aggregation)
        
        for group_name, metrics in grouped_metrics.items():
            if metrics:
                agg_metric = agg_func(metrics, group_name)
                if agg_metric:
                    aggregated_metrics.append(agg_metric)
        
        # Create aggregated collection
        return MetricCollection(
            collection_id=str(uuid.uuid4()),
            name=f"aggregated_{aggregation}",
            description=f"Aggregated metrics using {aggregation}",
            metrics=aggregated_metrics,
            tags={'aggregation_type': aggregation, 'source_count': str(len(collections))}
        )
    
    def _group_metrics(self, metrics: List[MetricValue], group_by: str) -> Dict[str, List[MetricValue]]:
        """Group metrics by specified attribute."""
        groups = {}
        
        for metric in metrics:
            if group_by == 'name':
                key = metric.name
            elif group_by == 'type':
                key = metric.metric_type.value
            elif group_by == 'scope':
                key = metric.scope.value
            elif group_by == 'source':
                key = metric.source or 'unknown'
            else:
                key = getattr(metric, group_by, 'unknown')
            
            if key not in groups:
                groups[key] = []
            groups[key].append(metric)
        
        return groups
    
    def _sum_aggregation(self, metrics: List[MetricValue], group_name: str) -> Optional[MetricValue]:
        """Sum aggregation."""
        numeric_values = [m.value for m in metrics if isinstance(m.value, (int, float))]
        if not numeric_values:
            return None
        
        return MetricValue(
            name=f"{group_name}_sum",
            value=sum(numeric_values),
            unit=metrics[0].unit if metrics else None,
            metadata={'source_count': len(metrics), 'aggregation': 'sum'}
        )
    
    def _avg_aggregation(self, metrics: List[MetricValue], group_name: str) -> Optional[MetricValue]:
        """Average aggregation."""
        numeric_values = [m.value for m in metrics if isinstance(m.value, (int, float))]
        if not numeric_values:
            return None
        
        return MetricValue(
            name=f"{group_name}_avg",
            value=sum(numeric_values) / len(numeric_values),
            unit=metrics[0].unit if metrics else None,
            metadata={'source_count': len(metrics), 'aggregation': 'avg'}
        )
    
    def _min_aggregation(self, metrics: List[MetricValue], group_name: str) -> Optional[MetricValue]:
        """Minimum aggregation."""
        numeric_values = [m.value for m in metrics if isinstance(m.value, (int, float))]
        if not numeric_values:
            return None
        
        return MetricValue(
            name=f"{group_name}_min",
            value=min(numeric_values),
            unit=metrics[0].unit if metrics else None,
            metadata={'source_count': len(metrics), 'aggregation': 'min'}
        )
    
    def _max_aggregation(self, metrics: List[MetricValue], group_name: str) -> Optional[MetricValue]:
        """Maximum aggregation."""
        numeric_values = [m.value for m in metrics if isinstance(m.value, (int, float))]
        if not numeric_values:
            return None
        
        return MetricValue(
            name=f"{group_name}_max",
            value=max(numeric_values),
            unit=metrics[0].unit if metrics else None,
            metadata={'source_count': len(metrics), 'aggregation': 'max'}
        )
    
    def _count_aggregation(self, metrics: List[MetricValue], group_name: str) -> MetricValue:
        """Count aggregation."""
        return MetricValue(
            name=f"{group_name}_count",
            value=len(metrics),
            unit="count",
            metadata={'aggregation': 'count'}
        )
    
    def _std_aggregation(self, metrics: List[MetricValue], group_name: str) -> Optional[MetricValue]:
        """Standard deviation aggregation."""
        numeric_values = [m.value for m in metrics if isinstance(m.value, (int, float))]
        if len(numeric_values) < 2:
            return None
        
        mean = sum(numeric_values) / len(numeric_values)
        variance = sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values)
        std_dev = variance ** 0.5
        
        return MetricValue(
            name=f"{group_name}_std",
            value=std_dev,
            unit=metrics[0].unit if metrics else None,
            metadata={'source_count': len(metrics), 'aggregation': 'std', 'mean': mean}
        )


class MetricsExporter:
    """Export metrics to various formats and destinations."""
    
    def __init__(self):
        self.exporters = {
            'json': self._export_json,
            'csv': self._export_csv,
            'sqlite': self._export_sqlite,
            'prometheus': self._export_prometheus
        }
    
    def export_collection(self, 
                         collection: MetricCollection, 
                         format: str = 'json',
                         destination: Optional[str] = None) -> Union[str, bool]:
        """Export metric collection to specified format."""
        
        if format not in self.exporters:
            raise ValueError(f"Unsupported export format: {format}")
        
        exporter = self.exporters[format]
        return exporter(collection, destination)
    
    def export_collections(self,
                          collections: List[MetricCollection],
                          format: str = 'json',
                          destination: Optional[str] = None) -> Union[str, bool]:
        """Export multiple collections."""
        
        if format == 'json':
            data = {
                'export_timestamp': time.time(),
                'collection_count': len(collections),
                'collections': [c.to_dict() for c in collections]
            }
            
            if destination:
                with open(destination, 'w') as f:
                    json.dump(data, f, indent=2)
                return True
            else:
                return json.dumps(data, indent=2)
        
        elif format == 'sqlite':
            return self._export_collections_sqlite(collections, destination)
        
        else:
            # For other formats, export each collection separately
            results = []
            for i, collection in enumerate(collections):
                if destination:
                    dest_path = f"{destination}_{i}.{format}"
                else:
                    dest_path = None
                
                result = self.export_collection(collection, format, dest_path)
                results.append(result)
            
            return results
    
    def _export_json(self, collection: MetricCollection, destination: Optional[str]) -> Union[str, bool]:
        """Export to JSON format."""
        data = collection.to_dict()
        
        if destination:
            with open(destination, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        else:
            return json.dumps(data, indent=2)
    
    def _export_csv(self, collection: MetricCollection, destination: Optional[str]) -> Union[str, bool]:
        """Export to CSV format."""
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['name', 'value', 'unit', 'timestamp', 'type', 'scope', 'confidence', 'source'])
        
        # Write metrics
        for metric in collection.metrics:
            writer.writerow([
                metric.name,
                metric.value,
                metric.unit or '',
                metric.timestamp,
                metric.metric_type.value,
                metric.scope.value,
                metric.confidence or '',
                metric.source or ''
            ])
        
        csv_content = output.getvalue()
        output.close()
        
        if destination:
            with open(destination, 'w') as f:
                f.write(csv_content)
            return True
        else:
            return csv_content
    
    def _export_sqlite(self, collection: MetricCollection, destination: Optional[str]) -> bool:
        """Export to SQLite database."""
        if not destination:
            destination = ':memory:'
        
        conn = sqlite3.connect(destination)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collections (
                collection_id TEXT PRIMARY KEY,
                name TEXT,
                description TEXT,
                created_at REAL,
                tags TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_id TEXT,
                name TEXT,
                value TEXT,
                unit TEXT,
                timestamp REAL,
                metric_type TEXT,
                scope TEXT,
                confidence REAL,
                source TEXT,
                metadata TEXT,
                FOREIGN KEY (collection_id) REFERENCES collections (collection_id)
            )
        ''')
        
        # Insert collection
        cursor.execute('''
            INSERT OR REPLACE INTO collections 
            (collection_id, name, description, created_at, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            collection.collection_id,
            collection.name,
            collection.description,
            collection.created_at,
            json.dumps(collection.tags),
            json.dumps(collection.metadata)
        ))
        
        # Insert metrics
        for metric in collection.metrics:
            cursor.execute('''
                INSERT INTO metrics 
                (collection_id, name, value, unit, timestamp, metric_type, scope, confidence, source, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                collection.collection_id,
                metric.name,
                str(metric.value),
                metric.unit,
                metric.timestamp,
                metric.metric_type.value,
                metric.scope.value,
                metric.confidence,
                metric.source,
                json.dumps(metric.metadata)
            ))
        
        conn.commit()
        conn.close()
        return True
    
    def _export_collections_sqlite(self, collections: List[MetricCollection], destination: Optional[str]) -> bool:
        """Export multiple collections to SQLite."""
        if not destination:
            destination = ':memory:'
        
        for collection in collections:
            self._export_sqlite(collection, destination)
        
        return True
    
    def _export_prometheus(self, collection: MetricCollection, destination: Optional[str]) -> Union[str, bool]:
        """Export to Prometheus format."""
        lines = []
        
        for metric in collection.metrics:
            if isinstance(metric.value, (int, float)):
                # Prometheus metric name (replace invalid chars)
                metric_name = metric.name.replace('-', '_').replace('.', '_')
                
                # Add labels
                labels = []
                if metric.source:
                    labels.append(f'source="{metric.source}"')
                if metric.unit:
                    labels.append(f'unit="{metric.unit}"')
                labels.append(f'type="{metric.metric_type.value}"')
                labels.append(f'scope="{metric.scope.value}"')
                
                label_str = '{' + ','.join(labels) + '}' if labels else ''
                
                # Format metric line
                lines.append(f'{metric_name}{label_str} {metric.value} {int(metric.timestamp * 1000)}')
        
        prometheus_content = '\n'.join(lines)
        
        if destination:
            with open(destination, 'w') as f:
                f.write(prometheus_content)
            return True
        else:
            return prometheus_content


@dataclass
class MetricsConfiguration:
    """Configuration for metrics collection."""
    enabled_collectors: List[str] = field(default_factory=list)
    collection_interval: float = 10.0  # seconds
    retention_days: int = 30
    export_formats: List[str] = field(default_factory=lambda: ['json'])
    export_destinations: Dict[str, str] = field(default_factory=dict)
    aggregation_rules: Dict[str, Any] = field(default_factory=dict)
    alert_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetricsConfiguration':
        """Create from dictionary."""
        return cls(**data)
    
    def save_to_file(self, filepath: str):
        """Save configuration to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'MetricsConfiguration':
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class MetricsManager:
    """Main manager for comprehensive metrics collection and analysis."""
    
    def __init__(self, config: Optional[MetricsConfiguration] = None):
        self.config = config or MetricsConfiguration()
        self.registry = MetricsRegistry()
        self.aggregator = MetricsAggregator()
        self.exporter = MetricsExporter()
        
        self.collection_thread = None
        self.collection_active = False
        self.collected_metrics = []
        
    def start_collection(self):
        """Start automated metrics collection."""
        if self.collection_active:
            return
        
        self.collection_active = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info(f"Started metrics collection with {self.config.collection_interval}s interval")
    
    def stop_collection(self):
        """Stop automated metrics collection."""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=10)
        
        logger.info("Stopped metrics collection")
    
    def collect_manual(self, context: Optional[Dict[str, Any]] = None) -> List[MetricCollection]:
        """Manually trigger metrics collection."""
        context = context or {}
        collections = self.registry.collect_all_metrics(context)
        self.collected_metrics.extend(collections)
        
        # Apply retention policy
        self._apply_retention_policy()
        
        return collections
    
    def get_aggregated_metrics(self, 
                              timeframe_hours: Optional[int] = None,
                              group_by: Optional[str] = None,
                              aggregation: str = 'avg') -> MetricCollection:
        """Get aggregated metrics for specified timeframe."""
        
        if timeframe_hours:
            cutoff_time = time.time() - (timeframe_hours * 3600)
            relevant_collections = [
                c for c in self.collected_metrics 
                if c.created_at >= cutoff_time
            ]
        else:
            relevant_collections = self.collected_metrics
        
        return self.aggregator.aggregate_collections(relevant_collections, group_by, aggregation)
    
    def export_metrics(self,
                      collections: Optional[List[MetricCollection]] = None,
                      format: str = 'json',
                      destination: Optional[str] = None) -> Union[str, bool]:
        """Export metrics to specified format."""
        
        if collections is None:
            collections = self.collected_metrics
        
        return self.exporter.export_collections(collections, format, destination)
    
    def _collection_loop(self):
        """Main collection loop."""
        while self.collection_active:
            try:
                self.collect_manual()
                time.sleep(self.config.collection_interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                time.sleep(self.config.collection_interval)
    
    def _apply_retention_policy(self):
        """Apply retention policy to collected metrics."""
        if self.config.retention_days <= 0:
            return
        
        cutoff_time = time.time() - (self.config.retention_days * 24 * 3600)
        self.collected_metrics = [
            c for c in self.collected_metrics 
            if c.created_at >= cutoff_time
        ]