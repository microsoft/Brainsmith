"""
Historical Analysis Engine
Trend analysis, regression detection, baseline management, and intelligent alerting.
"""

import os
import sys
import time
import logging
import threading
import sqlite3
import json
import statistics
import math
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import pickle

from .core import MetricsCollector, MetricValue, MetricCollection, MetricType, MetricScope

logger = logging.getLogger(__name__)


@dataclass
class TrendPoint:
    """A single point in a trend analysis."""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Results of trend analysis."""
    metric_name: str
    time_period_hours: float
    data_points: int
    trend_direction: str  # 'increasing', 'decreasing', 'stable', 'volatile'
    trend_strength: float  # 0.0 to 1.0
    slope: float
    correlation_coefficient: float
    mean_value: float
    std_deviation: float
    min_value: float
    max_value: float
    volatility: float
    recent_change_percent: float
    predictions: Dict[str, float] = field(default_factory=dict)


@dataclass
class RegressionDetection:
    """Results of regression detection."""
    metric_name: str
    regression_detected: bool
    severity: str  # 'minor', 'moderate', 'severe'
    baseline_value: float
    current_value: float
    percent_change: float
    detection_confidence: float
    detection_timestamp: float
    contributing_factors: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)


@dataclass
class Baseline:
    """Performance baseline information."""
    baseline_id: str
    name: str
    description: str
    created_at: float
    metrics: Dict[str, float]
    configuration: Dict[str, Any]
    tags: Dict[str, str] = field(default_factory=dict)
    is_active: bool = True


@dataclass
class Alert:
    """Performance alert information."""
    alert_id: str
    metric_name: str
    alert_type: str  # 'regression', 'threshold', 'anomaly'
    severity: str  # 'info', 'warning', 'error', 'critical'
    message: str
    current_value: float
    expected_value: Optional[float]
    threshold_value: Optional[float]
    timestamp: float
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsDatabase:
    """Database for storing and querying metrics history."""
    
    def __init__(self, db_path: str = "metrics_history.db"):
        self.db_path = db_path
        self.connection_pool = {}
        self.lock = threading.Lock()
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_id TEXT,
                    name TEXT,
                    value REAL,
                    unit TEXT,
                    timestamp REAL,
                    metric_type TEXT,
                    scope TEXT,
                    confidence REAL,
                    source TEXT,
                    metadata TEXT
                )
            ''')
            
            # Collections table
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
            
            # Baselines table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS baselines (
                    baseline_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    created_at REAL,
                    metrics TEXT,
                    configuration TEXT,
                    tags TEXT,
                    is_active INTEGER
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    metric_name TEXT,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    current_value REAL,
                    expected_value REAL,
                    threshold_value REAL,
                    timestamp REAL,
                    acknowledged INTEGER,
                    resolved INTEGER,
                    metadata TEXT
                )
            ''')
            
            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON metrics(name, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_type_timestamp ON metrics(metric_type, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
            
            conn.commit()
    
    def store_collection(self, collection: MetricCollection):
        """Store metric collection in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Store collection
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
            
            # Store metrics
            for metric in collection.metrics:
                cursor.execute('''
                    INSERT INTO metrics 
                    (collection_id, name, value, unit, timestamp, metric_type, scope, confidence, source, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    collection.collection_id,
                    metric.name,
                    float(metric.value) if isinstance(metric.value, (int, float)) else 0.0,
                    metric.unit,
                    metric.timestamp,
                    metric.metric_type.value,
                    metric.scope.value,
                    metric.confidence,
                    metric.source,
                    json.dumps(metric.metadata)
                ))
            
            conn.commit()
    
    def get_metric_history(self, 
                          metric_name: str,
                          hours: Optional[int] = None,
                          limit: Optional[int] = None) -> List[TrendPoint]:
        """Get historical data for a specific metric."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = 'SELECT timestamp, value, metadata FROM metrics WHERE name = ?'
            params = [metric_name]
            
            if hours:
                cutoff_time = time.time() - (hours * 3600)
                query += ' AND timestamp >= ?'
                params.append(cutoff_time)
            
            query += ' ORDER BY timestamp DESC'
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            trend_points = []
            for timestamp, value, metadata_json in rows:
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except:
                    metadata = {}
                
                trend_points.append(TrendPoint(
                    timestamp=timestamp,
                    value=value,
                    metadata=metadata
                ))
            
            return list(reversed(trend_points))  # Return in chronological order
    
    def get_metrics_by_type(self, 
                           metric_type: MetricType,
                           hours: Optional[int] = None) -> Dict[str, List[TrendPoint]]:
        """Get metrics grouped by name for a specific type."""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = 'SELECT name, timestamp, value, metadata FROM metrics WHERE metric_type = ?'
            params = [metric_type.value]
            
            if hours:
                cutoff_time = time.time() - (hours * 3600)
                query += ' AND timestamp >= ?'
                params.append(cutoff_time)
            
            query += ' ORDER BY name, timestamp'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            metrics_by_name = {}
            for name, timestamp, value, metadata_json in rows:
                if name not in metrics_by_name:
                    metrics_by_name[name] = []
                
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except:
                    metadata = {}
                
                metrics_by_name[name].append(TrendPoint(
                    timestamp=timestamp,
                    value=value,
                    metadata=metadata
                ))
            
            return metrics_by_name
    
    def store_baseline(self, baseline: Baseline):
        """Store performance baseline."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO baselines 
                (baseline_id, name, description, created_at, metrics, configuration, tags, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                baseline.baseline_id,
                baseline.name,
                baseline.description,
                baseline.created_at,
                json.dumps(baseline.metrics),
                json.dumps(baseline.configuration),
                json.dumps(baseline.tags),
                1 if baseline.is_active else 0
            ))
            
            conn.commit()
    
    def get_active_baseline(self) -> Optional[Baseline]:
        """Get the currently active baseline."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT baseline_id, name, description, created_at, metrics, configuration, tags
                FROM baselines 
                WHERE is_active = 1 
                ORDER BY created_at DESC 
                LIMIT 1
            ''')
            
            row = cursor.fetchone()
            if not row:
                return None
            
            baseline_id, name, description, created_at, metrics_json, config_json, tags_json = row
            
            return Baseline(
                baseline_id=baseline_id,
                name=name,
                description=description,
                created_at=created_at,
                metrics=json.loads(metrics_json),
                configuration=json.loads(config_json),
                tags=json.loads(tags_json) if tags_json else {},
                is_active=True
            )
    
    def store_alert(self, alert: Alert):
        """Store performance alert."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO alerts 
                (alert_id, metric_name, alert_type, severity, message, current_value, 
                 expected_value, threshold_value, timestamp, acknowledged, resolved, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.metric_name,
                alert.alert_type,
                alert.severity,
                alert.message,
                alert.current_value,
                alert.expected_value,
                alert.threshold_value,
                alert.timestamp,
                1 if alert.acknowledged else 0,
                1 if alert.resolved else 0,
                json.dumps(alert.metadata)
            ))
            
            conn.commit()
    
    def get_recent_alerts(self, hours: int = 24, unresolved_only: bool = True) -> List[Alert]:
        """Get recent alerts."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cutoff_time = time.time() - (hours * 3600)
            query = 'SELECT * FROM alerts WHERE timestamp >= ?'
            params = [cutoff_time]
            
            if unresolved_only:
                query += ' AND resolved = 0'
            
            query += ' ORDER BY timestamp DESC'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            alerts = []
            for row in rows:
                (alert_id, metric_name, alert_type, severity, message, current_value,
                 expected_value, threshold_value, timestamp, acknowledged, resolved, metadata_json) = row
                
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except:
                    metadata = {}
                
                alerts.append(Alert(
                    alert_id=alert_id,
                    metric_name=metric_name,
                    alert_type=alert_type,
                    severity=severity,
                    message=message,
                    current_value=current_value,
                    expected_value=expected_value,
                    threshold_value=threshold_value,
                    timestamp=timestamp,
                    acknowledged=bool(acknowledged),
                    resolved=bool(resolved),
                    metadata=metadata
                ))
            
            return alerts


class TrendAnalyzer:
    """Analyze trends in performance metrics."""
    
    def __init__(self, min_data_points: int = 5):
        self.min_data_points = min_data_points
    
    def analyze_trend(self, 
                     metric_name: str,
                     trend_points: List[TrendPoint],
                     time_period_hours: float) -> TrendAnalysis:
        """Analyze trend for a specific metric."""
        
        if len(trend_points) < self.min_data_points:
            return TrendAnalysis(
                metric_name=metric_name,
                time_period_hours=time_period_hours,
                data_points=len(trend_points),
                trend_direction='insufficient_data',
                trend_strength=0.0,
                slope=0.0,
                correlation_coefficient=0.0,
                mean_value=0.0,
                std_deviation=0.0,
                min_value=0.0,
                max_value=0.0,
                volatility=0.0,
                recent_change_percent=0.0
            )
        
        try:
            # Extract values and timestamps
            values = [point.value for point in trend_points]
            timestamps = [point.timestamp for point in trend_points]
            
            # Basic statistics
            mean_value = statistics.mean(values)
            std_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
            min_value = min(values)
            max_value = max(values)
            
            # Calculate trend using linear regression
            slope, correlation_coefficient = self._calculate_linear_trend(timestamps, values)
            
            # Determine trend direction and strength
            trend_direction = self._classify_trend_direction(slope, correlation_coefficient)
            trend_strength = abs(correlation_coefficient)
            
            # Calculate volatility (coefficient of variation)
            volatility = (std_deviation / abs(mean_value)) if mean_value != 0 else 0.0
            
            # Calculate recent change percentage
            recent_change_percent = self._calculate_recent_change(values)
            
            # Generate predictions
            predictions = self._generate_predictions(timestamps, values, slope)
            
            return TrendAnalysis(
                metric_name=metric_name,
                time_period_hours=time_period_hours,
                data_points=len(trend_points),
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                slope=slope,
                correlation_coefficient=correlation_coefficient,
                mean_value=mean_value,
                std_deviation=std_deviation,
                min_value=min_value,
                max_value=max_value,
                volatility=volatility,
                recent_change_percent=recent_change_percent,
                predictions=predictions
            )
            
        except Exception as e:
            logger.error(f"Trend analysis failed for {metric_name}: {e}")
            return TrendAnalysis(
                metric_name=metric_name,
                time_period_hours=time_period_hours,
                data_points=len(trend_points),
                trend_direction='error',
                trend_strength=0.0,
                slope=0.0,
                correlation_coefficient=0.0,
                mean_value=0.0,
                std_deviation=0.0,
                min_value=0.0,
                max_value=0.0,
                volatility=0.0,
                recent_change_percent=0.0
            )
    
    def _calculate_linear_trend(self, timestamps: List[float], values: List[float]) -> Tuple[float, float]:
        """Calculate linear trend slope and correlation coefficient."""
        n = len(timestamps)
        
        # Normalize timestamps to start from 0
        min_timestamp = min(timestamps)
        x = [t - min_timestamp for t in timestamps]
        y = values
        
        # Calculate linear regression
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        # Slope
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            slope = 0.0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # Correlation coefficient
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
        
        if denominator == 0:
            correlation_coefficient = 0.0
        else:
            correlation_coefficient = numerator / denominator
        
        return slope, correlation_coefficient
    
    def _classify_trend_direction(self, slope: float, correlation: float) -> str:
        """Classify trend direction based on slope and correlation."""
        correlation_threshold = 0.3
        slope_threshold = 0.001
        
        if abs(correlation) < correlation_threshold:
            return 'stable'
        elif abs(slope) < slope_threshold:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _calculate_recent_change(self, values: List[float]) -> float:
        """Calculate recent change percentage (comparing last 25% vs previous 25%)."""
        if len(values) < 4:
            return 0.0
        
        quarter_size = len(values) // 4
        if quarter_size == 0:
            return 0.0
        
        recent_values = values[-quarter_size:]
        previous_values = values[-2*quarter_size:-quarter_size]
        
        if not previous_values:
            return 0.0
        
        recent_mean = statistics.mean(recent_values)
        previous_mean = statistics.mean(previous_values)
        
        if previous_mean == 0:
            return 0.0
        
        return ((recent_mean - previous_mean) / abs(previous_mean)) * 100
    
    def _generate_predictions(self, timestamps: List[float], values: List[float], slope: float) -> Dict[str, float]:
        """Generate simple predictions based on linear trend."""
        if not timestamps or not values:
            return {}
        
        last_timestamp = max(timestamps)
        last_value = values[timestamps.index(last_timestamp)]
        
        # Predict values for 1 hour, 6 hours, and 24 hours into the future
        predictions = {}
        for hours in [1, 6, 24]:
            future_timestamp = last_timestamp + (hours * 3600)
            time_delta = future_timestamp - last_timestamp
            predicted_value = last_value + (slope * time_delta)
            predictions[f"{hours}h"] = predicted_value
        
        return predictions


class RegressionDetector:
    """Detect performance regressions against baselines."""
    
    def __init__(self, 
                 minor_threshold: float = 5.0,
                 moderate_threshold: float = 15.0,
                 severe_threshold: float = 30.0):
        self.minor_threshold = minor_threshold
        self.moderate_threshold = moderate_threshold
        self.severe_threshold = severe_threshold
    
    def detect_regression(self,
                         metric_name: str,
                         current_value: float,
                         baseline: Baseline,
                         trend_analysis: Optional[TrendAnalysis] = None) -> Optional[RegressionDetection]:
        """Detect regression for a specific metric."""
        
        if metric_name not in baseline.metrics:
            return None
        
        baseline_value = baseline.metrics[metric_name]
        
        # Calculate percentage change
        if baseline_value == 0:
            if current_value == 0:
                percent_change = 0.0
            else:
                percent_change = float('inf')
        else:
            percent_change = ((current_value - baseline_value) / abs(baseline_value)) * 100
        
        # Determine if this is a regression based on metric type
        is_regression = self._is_regression(metric_name, percent_change)
        
        if not is_regression:
            return None
        
        # Determine severity
        abs_change = abs(percent_change)
        if abs_change >= self.severe_threshold:
            severity = 'severe'
        elif abs_change >= self.moderate_threshold:
            severity = 'moderate'
        elif abs_change >= self.minor_threshold:
            severity = 'minor'
        else:
            return None  # Below threshold
        
        # Calculate detection confidence
        confidence = self._calculate_detection_confidence(
            percent_change, trend_analysis, baseline_value, current_value
        )
        
        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(
            metric_name, percent_change, trend_analysis
        )
        
        # Generate recommended actions
        recommended_actions = self._generate_recommended_actions(
            metric_name, severity, percent_change, contributing_factors
        )
        
        return RegressionDetection(
            metric_name=metric_name,
            regression_detected=True,
            severity=severity,
            baseline_value=baseline_value,
            current_value=current_value,
            percent_change=percent_change,
            detection_confidence=confidence,
            detection_timestamp=time.time(),
            contributing_factors=contributing_factors,
            recommended_actions=recommended_actions
        )
    
    def _is_regression(self, metric_name: str, percent_change: float) -> bool:
        """Determine if change represents a regression based on metric type."""
        # Define metrics where decrease is bad (regression)
        decrease_bad_metrics = [
            'throughput', 'ops_per_sec', 'efficiency', 'performance',
            'bandwidth', 'utilization', 'accuracy', 'precision'
        ]
        
        # Define metrics where increase is bad (regression)
        increase_bad_metrics = [
            'latency', 'delay', 'power', 'error', 'failure',
            'time', 'duration', 'slack'  # negative slack is bad
        ]
        
        metric_lower = metric_name.lower()
        
        # Check if decrease is bad for this metric
        for bad_decrease_metric in decrease_bad_metrics:
            if bad_decrease_metric in metric_lower:
                return percent_change < -self.minor_threshold
        
        # Check if increase is bad for this metric
        for bad_increase_metric in increase_bad_metrics:
            if bad_increase_metric in metric_lower:
                return percent_change > self.minor_threshold
        
        # For unknown metrics, consider significant changes in either direction as potential regressions
        return abs(percent_change) > self.moderate_threshold
    
    def _calculate_detection_confidence(self,
                                      percent_change: float,
                                      trend_analysis: Optional[TrendAnalysis],
                                      baseline_value: float,
                                      current_value: float) -> float:
        """Calculate confidence in regression detection."""
        base_confidence = 0.7
        
        # Higher confidence for larger changes
        change_factor = min(abs(percent_change) / 50.0, 1.0)
        
        # Higher confidence if trend supports the regression
        trend_factor = 0.0
        if trend_analysis:
            if trend_analysis.trend_direction in ['decreasing', 'increasing']:
                trend_factor = trend_analysis.trend_strength * 0.3
        
        # Lower confidence for very small baseline values (more susceptible to noise)
        baseline_factor = 0.0
        if abs(baseline_value) < 1.0:
            baseline_factor = -0.2
        
        confidence = base_confidence + change_factor * 0.2 + trend_factor + baseline_factor
        return max(0.1, min(1.0, confidence))
    
    def _identify_contributing_factors(self,
                                     metric_name: str,
                                     percent_change: float,
                                     trend_analysis: Optional[TrendAnalysis]) -> List[str]:
        """Identify potential contributing factors to the regression."""
        factors = []
        
        # Trend-based factors
        if trend_analysis:
            if trend_analysis.volatility > 0.5:
                factors.append("High volatility in recent measurements")
            
            if trend_analysis.trend_direction in ['decreasing', 'increasing']:
                factors.append(f"Consistent {trend_analysis.trend_direction} trend detected")
            
            if trend_analysis.recent_change_percent > 20:
                factors.append("Significant recent change in metric values")
        
        # Magnitude-based factors
        if abs(percent_change) > 50:
            factors.append("Large magnitude change suggests systematic issue")
        elif abs(percent_change) > 20:
            factors.append("Moderate change may indicate configuration or algorithm modification")
        
        # Metric-specific factors
        if 'timing' in metric_name.lower() or 'latency' in metric_name.lower():
            factors.append("Possible causes: routing congestion, clock frequency, pipeline depth")
        elif 'throughput' in metric_name.lower() or 'performance' in metric_name.lower():
            factors.append("Possible causes: resource utilization, memory bandwidth, parallelism")
        elif 'power' in metric_name.lower():
            factors.append("Possible causes: clock frequency, resource usage, activity factor")
        
        return factors
    
    def _generate_recommended_actions(self,
                                    metric_name: str,
                                    severity: str,
                                    percent_change: float,
                                    contributing_factors: List[str]) -> List[str]:
        """Generate recommended actions to address the regression."""
        actions = []
        
        # Immediate actions based on severity
        if severity == 'severe':
            actions.append("URGENT: Investigate immediately and consider reverting recent changes")
            actions.append("Analyze build logs and synthesis reports for errors or warnings")
        elif severity == 'moderate':
            actions.append("Investigate root cause within 24 hours")
            actions.append("Compare current configuration with baseline")
        else:  # minor
            actions.append("Monitor trend and investigate if pattern continues")
        
        # Metric-specific actions
        if 'timing' in metric_name.lower():
            actions.extend([
                "Check timing closure in synthesis/implementation reports",
                "Analyze critical path and consider pipeline optimization",
                "Verify clock constraints and frequency settings"
            ])
        elif 'throughput' in metric_name.lower():
            actions.extend([
                "Analyze resource utilization and identify bottlenecks",
                "Check memory bandwidth and data access patterns",
                "Consider increasing parallelism or optimizing data flow"
            ])
        elif 'power' in metric_name.lower():
            actions.extend([
                "Check clock frequency and gating strategies",
                "Analyze resource usage and activity factors",
                "Consider power optimization techniques"
            ])
        elif 'utilization' in metric_name.lower():
            actions.extend([
                "Check resource allocation and mapping",
                "Analyze synthesis optimization settings",
                "Consider design partitioning or resource sharing"
            ])
        
        # General actions
        actions.extend([
            "Document regression in issue tracking system",
            "Create new baseline if regression is intended/acceptable",
            "Set up enhanced monitoring for this metric"
        ])
        
        return actions


class BaselineManager:
    """Manage performance baselines."""
    
    def __init__(self, database: MetricsDatabase):
        self.database = database
    
    def create_baseline(self,
                       name: str,
                       description: str,
                       metrics: Dict[str, float],
                       configuration: Dict[str, Any],
                       tags: Optional[Dict[str, str]] = None,
                       set_as_active: bool = True) -> Baseline:
        """Create a new performance baseline."""
        
        baseline = Baseline(
            baseline_id=f"baseline_{int(time.time())}",
            name=name,
            description=description,
            created_at=time.time(),
            metrics=metrics.copy(),
            configuration=configuration.copy(),
            tags=tags or {},
            is_active=set_as_active
        )
        
        # Deactivate previous baseline if this one is being set as active
        if set_as_active:
            self._deactivate_current_baseline()
        
        # Store baseline
        self.database.store_baseline(baseline)
        
        logger.info(f"Created baseline '{name}' with {len(metrics)} metrics")
        return baseline
    
    def create_baseline_from_collection(self,
                                       collection: MetricCollection,
                                       name: Optional[str] = None,
                                       description: Optional[str] = None,
                                       set_as_active: bool = True) -> Baseline:
        """Create baseline from a metric collection."""
        
        # Extract metrics from collection
        metrics = {}
        for metric in collection.metrics:
            if isinstance(metric.value, (int, float)):
                metrics[metric.name] = float(metric.value)
        
        # Use collection metadata as configuration
        configuration = collection.metadata.copy()
        configuration['collection_id'] = collection.collection_id
        configuration['collection_created_at'] = collection.created_at
        
        return self.create_baseline(
            name=name or f"Baseline from {collection.name}",
            description=description or f"Baseline created from collection {collection.collection_id}",
            metrics=metrics,
            configuration=configuration,
            tags=collection.tags,
            set_as_active=set_as_active
        )
    
    def get_active_baseline(self) -> Optional[Baseline]:
        """Get the currently active baseline."""
        return self.database.get_active_baseline()
    
    def _deactivate_current_baseline(self):
        """Deactivate the currently active baseline."""
        current_baseline = self.database.get_active_baseline()
        if current_baseline:
            current_baseline.is_active = False
            self.database.store_baseline(current_baseline)


class AlertSystem:
    """Intelligent alerting system for performance metrics."""
    
    def __init__(self, database: MetricsDatabase):
        self.database = database
        self.alert_callbacks = []
        self.threshold_config = {}
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def set_threshold(self, metric_name: str, threshold_value: float, comparison: str = 'greater'):
        """Set threshold for a metric."""
        self.threshold_config[metric_name] = {
            'threshold': threshold_value,
            'comparison': comparison  # 'greater', 'less', 'absolute'
        }
    
    def check_thresholds(self, collection: MetricCollection):
        """Check metrics against configured thresholds."""
        for metric in collection.metrics:
            if metric.name in self.threshold_config:
                config = self.threshold_config[metric.name]
                
                alert = self._check_threshold(metric, config)
                if alert:
                    self._emit_alert(alert)
    
    def check_regressions(self, 
                         collection: MetricCollection,
                         regression_detections: List[RegressionDetection]):
        """Check for regressions and generate alerts."""
        for regression in regression_detections:
            alert = Alert(
                alert_id=f"regression_{int(time.time())}_{regression.metric_name}",
                metric_name=regression.metric_name,
                alert_type='regression',
                severity=self._map_regression_severity(regression.severity),
                message=f"Performance regression detected in {regression.metric_name}: "
                       f"{regression.percent_change:.1f}% change from baseline",
                current_value=regression.current_value,
                expected_value=regression.baseline_value,
                threshold_value=None,
                timestamp=time.time(),
                metadata={
                    'regression_severity': regression.severity,
                    'percent_change': regression.percent_change,
                    'detection_confidence': regression.detection_confidence,
                    'contributing_factors': regression.contributing_factors,
                    'recommended_actions': regression.recommended_actions
                }
            )
            
            self._emit_alert(alert)
    
    def _check_threshold(self, metric: MetricValue, config: Dict[str, Any]) -> Optional[Alert]:
        """Check if metric violates threshold."""
        if not isinstance(metric.value, (int, float)):
            return None
        
        threshold = config['threshold']
        comparison = config['comparison']
        
        violation = False
        if comparison == 'greater' and metric.value > threshold:
            violation = True
        elif comparison == 'less' and metric.value < threshold:
            violation = True
        elif comparison == 'absolute' and abs(metric.value) > threshold:
            violation = True
        
        if not violation:
            return None
        
        # Determine severity based on how much threshold is exceeded
        excess_ratio = abs(metric.value - threshold) / abs(threshold) if threshold != 0 else 1.0
        
        if excess_ratio > 0.5:
            severity = 'critical'
        elif excess_ratio > 0.2:
            severity = 'error'
        elif excess_ratio > 0.1:
            severity = 'warning'
        else:
            severity = 'info'
        
        return Alert(
            alert_id=f"threshold_{int(time.time())}_{metric.name}",
            metric_name=metric.name,
            alert_type='threshold',
            severity=severity,
            message=f"Metric {metric.name} exceeded threshold: {metric.value} {comparison} {threshold}",
            current_value=float(metric.value),
            expected_value=None,
            threshold_value=threshold,
            timestamp=time.time(),
            metadata={
                'comparison': comparison,
                'excess_ratio': excess_ratio,
                'metric_unit': metric.unit
            }
        )
    
    def _map_regression_severity(self, regression_severity: str) -> str:
        """Map regression severity to alert severity."""
        mapping = {
            'minor': 'warning',
            'moderate': 'error',
            'severe': 'critical'
        }
        return mapping.get(regression_severity, 'warning')
    
    def _emit_alert(self, alert: Alert):
        """Emit alert to callbacks and store in database."""
        # Store in database
        self.database.store_alert(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        # Log alert
        logger.warning(f"ALERT [{alert.severity.upper()}] {alert.metric_name}: {alert.message}")


class HistoricalAnalysisEngine(MetricsCollector):
    """Main historical analysis engine that implements MetricsCollector interface."""
    
    def __init__(self, name: str = "HistoricalAnalysisEngine", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        
        self.database = MetricsDatabase(config.get('db_path', 'metrics_history.db') if config else 'metrics_history.db')
        self.trend_analyzer = TrendAnalyzer()
        self.regression_detector = RegressionDetector()
        self.baseline_manager = BaselineManager(self.database)
        self.alert_system = AlertSystem(self.database)
        
        # Configuration
        self.analysis_window_hours = config.get('analysis_window_hours', 24) if config else 24
        self.auto_baseline_creation = config.get('auto_baseline_creation', False) if config else False
    
    def collect_metrics(self, context: Dict[str, Any]) -> MetricCollection:
        """Collect historical analysis metrics."""
        
        collection = MetricCollection(
            collection_id=f"historical_{int(time.time())}",
            name="Historical Analysis Metrics",
            description="Trend analysis, regression detection, and baseline comparison"
        )
        
        try:
            # Store current metrics collection in database for future analysis
            current_collection = context.get('current_collection')
            if current_collection:
                self.database.store_collection(current_collection)
                
                # Check thresholds
                self.alert_system.check_thresholds(current_collection)
                
                # Auto-create baseline if configured
                if self.auto_baseline_creation and not self.baseline_manager.get_active_baseline():
                    self.baseline_manager.create_baseline_from_collection(
                        current_collection,
                        name="Auto-generated baseline",
                        description="Automatically created baseline from first collection"
                    )
            
            # Perform trend analysis for key metrics
            key_metrics = context.get('key_metrics', [
                'throughput_ops_per_sec', 'latency_cycles', 'power_total_mw',
                'lut_utilization', 'dsp_utilization', 'timing_slack'
            ])
            
            trend_analyses = {}
            regression_detections = []
            
            for metric_name in key_metrics:
                # Get historical data
                trend_points = self.database.get_metric_history(
                    metric_name, hours=self.analysis_window_hours
                )
                
                if len(trend_points) >= 2:
                    # Perform trend analysis
                    trend_analysis = self.trend_analyzer.analyze_trend(
                        metric_name, trend_points, self.analysis_window_hours
                    )
                    trend_analyses[metric_name] = trend_analysis
                    
                    # Check for regressions
                    baseline = self.baseline_manager.get_active_baseline()
                    if baseline and len(trend_points) > 0:
                        current_value = trend_points[-1].value
                        regression = self.regression_detector.detect_regression(
                            metric_name, current_value, baseline, trend_analysis
                        )
                        if regression:
                            regression_detections.append(regression)
            
            # Generate alerts for regressions
            if regression_detections:
                self.alert_system.check_regressions(current_collection, regression_detections)
            
            # Add analysis results as metrics
            self._add_analysis_metrics(collection, trend_analyses, regression_detections)
            
        except Exception as e:
            logger.error(f"Failed to collect historical analysis metrics: {e}")
            collection.add_metric(MetricValue(
                name="analysis_error",
                value=str(e),
                metric_type=MetricType.PERFORMANCE,
                scope=MetricScope.GLOBAL
            ))
        
        return collection
    
    def get_supported_metrics(self) -> List[str]:
        """Get list of supported metrics."""
        return [
            "trend_direction", "trend_strength", "volatility",
            "regression_count", "alert_count", "baseline_age"
        ]
    
    def _add_analysis_metrics(self,
                            collection: MetricCollection,
                            trend_analyses: Dict[str, TrendAnalysis],
                            regression_detections: List[RegressionDetection]):
        """Add analysis results as metrics."""
        
        # Summary metrics
        collection.add_metric(MetricValue(
            "analyzed_metrics_count",
            len(trend_analyses),
            "count",
            metric_type=MetricType.PERFORMANCE
        ))
        
        collection.add_metric(MetricValue(
            "regression_count",
            len(regression_detections),
            "count",
            metric_type=MetricType.QUALITY
        ))
        
        # Recent alerts count
        recent_alerts = self.database.get_recent_alerts(hours=24)
        collection.add_metric(MetricValue(
            "recent_alerts_count",
            len(recent_alerts),
            "count",
            metric_type=MetricType.QUALITY
        ))
        
        # Baseline age
        baseline = self.baseline_manager.get_active_baseline()
        if baseline:
            baseline_age_hours = (time.time() - baseline.created_at) / 3600
            collection.add_metric(MetricValue(
                "baseline_age_hours",
                baseline_age_hours,
                "hours",
                metric_type=MetricType.PERFORMANCE
            ))
        
        # Individual trend metrics
        for metric_name, trend in trend_analyses.items():
            # Add trend metrics with metric name prefix
            collection.add_metric(MetricValue(
                f"{metric_name}_trend_strength",
                trend.trend_strength,
                "ratio",
                metric_type=MetricType.PERFORMANCE
            ))
            
            collection.add_metric(MetricValue(
                f"{metric_name}_volatility",
                trend.volatility,
                "ratio",
                metric_type=MetricType.PERFORMANCE
            ))
            
            collection.add_metric(MetricValue(
                f"{metric_name}_recent_change_percent",
                trend.recent_change_percent,
                "%",
                metric_type=MetricType.PERFORMANCE
            ))
        
        # Store detailed analysis in metadata
        collection.metadata['trend_analyses'] = {
            name: {
                'trend_direction': trend.trend_direction,
                'trend_strength': trend.trend_strength,
                'volatility': trend.volatility,
                'recent_change_percent': trend.recent_change_percent,
                'predictions': trend.predictions
            }
            for name, trend in trend_analyses.items()
        }
        
        collection.metadata['regression_detections'] = [
            {
                'metric_name': reg.metric_name,
                'severity': reg.severity,
                'percent_change': reg.percent_change,
                'detection_confidence': reg.detection_confidence
            }
            for reg in regression_detections
        ]
    
    def get_trend_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of trends for all metrics."""
        # Get metrics by type
        performance_metrics = self.database.get_metrics_by_type(MetricType.PERFORMANCE, hours)
        resource_metrics = self.database.get_metrics_by_type(MetricType.RESOURCE, hours)
        
        all_metrics = {**performance_metrics, **resource_metrics}
        
        summary = {
            'analysis_period_hours': hours,
            'metrics_analyzed': len(all_metrics),
            'trends': {}
        }
        
        for metric_name, trend_points in all_metrics.items():
            if len(trend_points) >= 2:
                trend = self.trend_analyzer.analyze_trend(metric_name, trend_points, hours)
                summary['trends'][metric_name] = {
                    'direction': trend.trend_direction,
                    'strength': trend.trend_strength,
                    'recent_change': trend.recent_change_percent
                }
        
        return summary
    
    def get_regression_summary(self) -> Dict[str, Any]:
        """Get summary of recent regressions."""
        recent_alerts = self.database.get_recent_alerts(hours=24, unresolved_only=True)
        regression_alerts = [alert for alert in recent_alerts if alert.alert_type == 'regression']
        
        summary = {
            'total_regressions': len(regression_alerts),
            'by_severity': {},
            'by_metric': {},
            'recent_regressions': []
        }
        
        # Group by severity
        for alert in regression_alerts:
            severity = alert.metadata.get('regression_severity', 'unknown')
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
        
        # Group by metric
        for alert in regression_alerts:
            metric = alert.metric_name
            summary['by_metric'][metric] = summary['by_metric'].get(metric, 0) + 1
        
        # Recent regressions details
        for alert in regression_alerts[:5]:  # Last 5
            summary['recent_regressions'].append({
                'metric_name': alert.metric_name,
                'severity': alert.metadata.get('regression_severity', 'unknown'),
                'percent_change': alert.metadata.get('percent_change', 0),
                'timestamp': alert.timestamp
            })
        
        return summary