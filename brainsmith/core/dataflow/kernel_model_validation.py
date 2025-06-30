############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Robust validation utilities for KernelModel configurations"""

from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from .kernel_model import KernelModel
from .kernel_definition import KernelDefinition
from .base import BaseModel
from .input_interface import InputInterface
from .output_interface import OutputInterface
from .relationships import RelationType


@dataclass
class ValidationIssue:
    """Represents a validation issue with severity and suggestions"""
    severity: str  # "error", "warning", "info"
    message: str
    interface: Optional[str] = None
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        prefix = f"[{self.interface}] " if self.interface else ""
        result = f"{self.severity.upper()}: {prefix}{self.message}"
        if self.suggestion:
            result += f"\n  Suggestion: {self.suggestion}"
        return result


class KernelModelValidator:
    """Provides comprehensive validation for KernelModel configurations"""
    
    def __init__(self, kernel_model: KernelModel):
        self.model = kernel_model
        self.definition = kernel_model.definition
        self.issues: List[ValidationIssue] = []
    
    def validate_configuration(self) -> List[ValidationIssue]:
        """Perform comprehensive validation of kernel configuration
        
        Returns:
            List of validation issues found
        """
        self.issues.clear()
        
        # Basic validation
        self._validate_interface_consistency()
        self._validate_dimension_compatibility()
        self._validate_parallelism_configuration()
        
        # Relationship validation
        if self.definition:
            self._validate_relationships()
            self._validate_parallelism_constraints()
        
        # Performance validation
        self._validate_performance_feasibility()
        
        return self.issues
    
    def _validate_interface_consistency(self):
        """Validate that interfaces match their definitions"""
        if not self.definition:
            return
            
        # Check all required interfaces are present
        required_input_names = {idef.name for idef in self.definition.input_definitions}
        required_output_names = {odef.name for odef in self.definition.output_definitions}
        required_names = required_input_names | required_output_names
        
        all_interfaces = self.model.input_models + self.model.output_models
        actual_names = {im.definition.name for im in all_interfaces 
                       if im.definition}
        
        missing = required_names - actual_names
        if missing:
            self.issues.append(ValidationIssue(
                severity="error",
                message=f"Missing required interfaces: {', '.join(missing)}",
                suggestion="Ensure all interfaces defined in KernelDefinition have corresponding models"
            ))
        
        extra = actual_names - required_names
        if extra:
            self.issues.append(ValidationIssue(
                severity="warning",
                message=f"Extra interfaces not in definition: {', '.join(extra)}",
                suggestion="Remove extra interfaces or update KernelDefinition"
            ))
    
    def _validate_dimension_compatibility(self):
        """Validate interface dimensions are compatible"""
        all_interfaces = self.model.input_models + self.model.output_models
        for intf in all_interfaces:
            if not intf.definition:
                continue
                
            # Check tensor dims match block dims length
            if len(intf.tensor_dims) != len(intf.block_dims):
                self.issues.append(ValidationIssue(
                    severity="error",
                    interface=intf.definition.name,
                    message=f"Tensor dims {intf.tensor_dims} and block dims {intf.block_dims} have different lengths",
                    suggestion="Ensure tensor and block dimensions have the same number of dimensions"
                ))
            
            # Check block divides tensor
            for i, (t, b) in enumerate(zip(intf.tensor_dims, intf.block_dims)):
                if t % b != 0:
                    self.issues.append(ValidationIssue(
                        severity="error",
                        interface=intf.definition.name,
                        message=f"Tensor dim {t} not divisible by block dim {b} at index {i}",
                        suggestion=f"Change block dim to a divisor of {t} (e.g., {self._suggest_divisors(t)})"
                    ))
            
            # Check SDIM compatibility with block dims
            if hasattr(intf, 'sdim') and intf.sdim:
                for i, (b, s) in enumerate(zip(intf.block_dims, intf.sdim)):
                    if s > b:
                        self.issues.append(ValidationIssue(
                            severity="error",
                            interface=intf.definition.name,
                            message=f"SDIM[{i}]={s} exceeds block dim {b}",
                            suggestion=f"Reduce SDIM[{i}] to at most {b}"
                        ))
    
    def _validate_parallelism_configuration(self):
        """Validate parallelism settings"""
        all_interfaces = self.model.input_models + self.model.output_models
        for intf in all_interfaces:
            if not intf.definition:
                continue
                
            # Check parallelism is reasonable
            # InputInterface has streaming_bandwidth, OutputInterface has streaming_rate
            bandwidth = None
            if hasattr(intf, 'streaming_bandwidth'):
                bandwidth = intf.streaming_bandwidth
            elif hasattr(intf, 'streaming_rate'):
                bandwidth = intf.streaming_rate
            
            if bandwidth and bandwidth > 1024:
                self.issues.append(ValidationIssue(
                    severity="warning",
                    interface=intf.definition.name,
                    message=f"Very high parallelism ({bandwidth}) may not be implementable",
                    suggestion="Consider reducing parallelism or verify hardware can support this level"
                ))
            
            # Check iPar vs block size
            block_size = 1
            for dim in intf.block_dims:
                block_size *= dim
            
            # Check bandwidth doesn't exceed block size
            bandwidth = None
            if hasattr(intf, 'streaming_bandwidth'):
                bandwidth = intf.streaming_bandwidth
            elif hasattr(intf, 'streaming_rate'):
                bandwidth = intf.streaming_rate
                
            if bandwidth and bandwidth > block_size:
                self.issues.append(ValidationIssue(
                    severity="error",
                    interface=intf.definition.name,
                    message=f"Streaming bandwidth ({bandwidth}) exceeds block size ({block_size})",
                    suggestion=f"Reduce parallelism to have total bandwidth at most {block_size}"
                ))
    
    def _validate_relationships(self):
        """Validate relationship constraints are satisfied"""
        if not self.definition:
            return
            
        context = {}
        all_interfaces = self.model.input_models + self.model.output_models
        for intf in all_interfaces:
            if intf.definition:
                context[intf.definition.name] = intf
        
        for rel in self.definition.relationships:
            source = context.get(rel.source_interface)
            target = context.get(rel.target_interface)
            
            if not source or not target:
                continue
                
            # Validate based on relationship type
            if rel.relation == RelationType.EQUAL:
                if not self._check_equal_relationship(source, target, rel):
                    self.issues.append(ValidationIssue(
                        severity="error",
                        message=f"EQUAL relationship violated between {rel.source_interface} and {rel.target_interface}",
                        suggestion=f"Ensure dimensions match: {rel.describe()}"
                    ))
            
            elif rel.relation == RelationType.MULTIPLE and rel.factor:
                if not self._check_multiple_relationship(source, target, rel):
                    self.issues.append(ValidationIssue(
                        severity="error",
                        message=f"MULTIPLE relationship violated: {rel.describe()}",
                        suggestion=f"Adjust dimensions to satisfy the {rel.factor}x relationship"
                    ))
    
    def _validate_parallelism_constraints(self):
        """Validate parallelism propagation constraints"""
        # Check for parallelism mismatches in related interfaces
        if not self.definition:
            return
            
        for rel in self.definition.relationships:
            # Find interfaces by name
            source = None
            target = None
            
            for inp in self.model.input_models:
                if inp.definition and inp.definition.name == rel.source_interface:
                    source = inp
                if inp.definition and inp.definition.name == rel.target_interface:
                    target = inp
            
            for out in self.model.output_models:
                if out.definition and out.definition.name == rel.source_interface:
                    source = out
                if out.definition and out.definition.name == rel.target_interface:
                    target = out
            
            if not source or not target:
                continue
                
            # For EQUAL relationships with matching dimensions, 
            # parallelism should be compatible
            if (rel.relation == RelationType.EQUAL and 
                rel.source_dim is not None and rel.target_dim is not None):
                
                source_blocks = source.block_dims
                target_blocks = target.block_dims
                
                if (rel.source_dim < len(source_blocks) and 
                    rel.target_dim < len(target_blocks)):
                    
                    source_dim = source_blocks[rel.source_dim]
                    target_dim = target_blocks[rel.target_dim]
                    
                    if source_dim == target_dim:
                        # Check parallelism compatibility
                        # For equal dimensions, stream dimensions should ideally match
                        source_stream = source.sdim[rel.source_dim] if hasattr(source, 'sdim') and source.sdim else 1
                        target_stream = target.sdim[rel.target_dim] if hasattr(target, 'sdim') and target.sdim else 1
                        
                        # Check if parallelism values differ significantly (more than 2x)
                        # Get bandwidth for source
                        if hasattr(source, 'streaming_bandwidth'):
                            source_bw = source.streaming_bandwidth
                        elif hasattr(source, 'streaming_rate'):
                            source_bw = source.streaming_rate
                        else:
                            source_bw = 0
                            
                        # Get bandwidth for target
                        if hasattr(target, 'streaming_bandwidth'):
                            target_bw = target.streaming_bandwidth
                        elif hasattr(target, 'streaming_rate'):
                            target_bw = target.streaming_rate
                        else:
                            target_bw = 0
                        
                        if source_bw > 1 and target_bw > 1:
                            ratio = max(source_bw, target_bw) / min(source_bw, target_bw)
                            if ratio >= 2:
                                self.issues.append(ValidationIssue(
                                    severity="warning",
                                    message=f"Parallelism mismatch on equal dimensions: {rel.source_interface}[{rel.source_dim}] has bandwidth={source_bw}, {rel.target_interface}[{rel.target_dim}] has bandwidth={target_bw}",
                                    suggestion="Consider using consistent parallelism values for related interfaces"
                                ))
    
    def _validate_performance_feasibility(self):
        """Validate performance metrics are feasible"""
        # Check bandwidth requirements
        try:
            metrics = self.model.calculate_performance_metrics()
            total_bw = metrics.get("aggregate", {}).get("total_bandwidth_mbps", 0)
            if total_bw > 100000:  # 100 GB/s
                self.issues.append(ValidationIssue(
                    severity="warning",
                    message=f"Very high total bandwidth requirement: {total_bw:.1f} MB/s",
                    suggestion="Consider reducing parallelism or using HBM/high-bandwidth memory"
                ))
        except Exception:
            # If performance metrics can't be calculated, skip this check
            pass
        
        # Check resource utilization
        if hasattr(self.model, 'resources') and self.model.resources:
            if self.model.resources.get("DSP", 0) > 10000:
                self.issues.append(ValidationIssue(
                    severity="warning", 
                    message=f"High DSP usage estimate: {self.model.resources['DSP']}",
                    suggestion="Verify target FPGA has sufficient DSP resources"
                ))
    
    def _check_equal_relationship(self, source: BaseModel, target: BaseModel,
                                 rel: 'DimensionRelationship') -> bool:
        """Check if EQUAL relationship is satisfied"""
        if rel.source_dim is None or rel.target_dim is None:
            # Total size equality
            source_size = 1
            target_size = 1
            for dim in source.tensor_dims:
                source_size *= dim
            for dim in target.tensor_dims:
                target_size *= dim
            return source_size == target_size
        else:
            # Specific dimension equality
            if (rel.source_dim < len(source.tensor_dims) and 
                rel.target_dim < len(target.tensor_dims)):
                return source.tensor_dims[rel.source_dim] == target.tensor_dims[rel.target_dim]
        return False
    
    def _check_multiple_relationship(self, source: BaseModel, target: BaseModel,
                                    rel: 'DimensionRelationship') -> bool:
        """Check if MULTIPLE relationship is satisfied"""
        if not rel.factor:
            return False
            
        if rel.source_dim is None or rel.target_dim is None:
            # Total size multiple
            source_size = 1
            target_size = 1
            for dim in source.tensor_dims:
                source_size *= dim
            for dim in target.tensor_dims:
                target_size *= dim
            return abs(source_size - rel.factor * target_size) < 0.001
        else:
            # Specific dimension multiple
            if (rel.source_dim < len(source.tensor_dims) and 
                rel.target_dim < len(target.tensor_dims)):
                return abs(source.tensor_dims[rel.source_dim] - 
                          rel.factor * target.tensor_dims[rel.target_dim]) < 0.001
        return False
    
    def _suggest_divisors(self, n: int) -> str:
        """Get string of divisors for suggestions"""
        divisors = []
        for i in range(1, min(n + 1, 10)):
            if n % i == 0:
                divisors.append(str(i))
        if n > 10:
            divisors.append("...")
        return ", ".join(divisors)
    
    def detect_conflicts(self) -> List[ValidationIssue]:
        """Detect configuration conflicts that might cause issues
        
        Returns:
            List of potential conflicts
        """
        conflicts = []
        
        # Check for interface name conflicts
        seen_names = set()
        all_interfaces = self.model.input_models + self.model.output_models
        for intf in all_interfaces:
            if intf.definition:
                if intf.definition.name in seen_names:
                    conflicts.append(ValidationIssue(
                        severity="error",
                        interface=intf.definition.name,
                        message="Duplicate interface name",
                        suggestion="Ensure all interfaces have unique names"
                    ))
                seen_names.add(intf.definition.name)
        
        # Check for circular dependencies in relationships
        if self.definition:
            cycles = self._detect_relationship_cycles()
            for cycle in cycles:
                conflicts.append(ValidationIssue(
                    severity="warning",
                    message=f"Circular dependency detected: {' -> '.join(cycle)}",
                    suggestion="Review relationships to remove circular dependencies"
                ))
        
        return conflicts
    
    def _detect_relationship_cycles(self) -> List[List[str]]:
        """Detect cycles in relationship graph"""
        if not self.definition:
            return []
            
        # Build adjacency list
        graph = {}
        for rel in self.definition.relationships:
            if rel.source_interface not in graph:
                graph[rel.source_interface] = []
            graph[rel.source_interface].append(rel.target_interface)
        
        # DFS to find cycles
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                dfs(node)
        
        return cycles
    
    def suggest_fixes(self) -> List[str]:
        """Generate suggestions for fixing validation issues
        
        Returns:
            List of actionable suggestions
        """
        suggestions = []
        
        # Analyze issues and generate fixes
        error_count = sum(1 for issue in self.issues if issue.severity == "error")
        warning_count = sum(1 for issue in self.issues if issue.severity == "warning")
        
        if error_count > 0:
            suggestions.append(f"Fix {error_count} errors before proceeding:")
            
            # Group errors by type
            dim_errors = [i for i in self.issues if "divisible" in i.message and i.severity == "error"]
            if dim_errors:
                suggestions.append("  - Adjust block dimensions to divide tensor dimensions evenly")
            
            par_errors = [i for i in self.issues if "iPar" in i.message and i.severity == "error"]
            if par_errors:
                suggestions.append("  - Reduce parallelism values to valid ranges")
        
        if warning_count > 0:
            suggestions.append(f"Consider addressing {warning_count} warnings:")
            
            perf_warnings = [i for i in self.issues if "bandwidth" in i.message or "DSP" in i.message]
            if perf_warnings:
                suggestions.append("  - Review performance requirements and hardware capabilities")
        
        return suggestions


def validate_kernel_model(model: KernelModel, verbose: bool = False) -> bool:
    """Convenience function to validate a kernel model
    
    Args:
        model: KernelModel to validate
        verbose: If True, print all issues
        
    Returns:
        True if no errors found
    """
    validator = KernelModelValidator(model)
    issues = validator.validate_configuration()
    conflicts = validator.detect_conflicts()
    
    all_issues = issues + conflicts
    errors = [i for i in all_issues if i.severity == "error"]
    
    if verbose and all_issues:
        print("Validation Report:")
        print("-" * 60)
        for issue in all_issues:
            print(issue)
        print("-" * 60)
        
        suggestions = validator.suggest_fixes()
        if suggestions:
            print("\nSuggested fixes:")
            for suggestion in suggestions:
                print(suggestion)
    
    return len(errors) == 0