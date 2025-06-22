############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Configuration management for design space exploration"""

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple, Union
import math
from ..core.kernel import Kernel
from ..core.interface import Interface
from ..core.types import Shape, InterfaceDirection, prod


@dataclass
class ParallelismConfig:
    """Parallelism configuration for kernels
    
    Defines how to parallelize interfaces across a dataflow graph.
    Supports both uniform and per-interface parallelism settings.
    """
    
    # Per-interface parallelism: (kernel_name, interface_name) -> parallelism
    interface_pars: Dict[Tuple[str, str], int] = field(default_factory=dict)
    
    # Global parallelism (applied if interface not in interface_pars)
    global_par: Optional[int] = None
    
    # Resource constraints
    max_bandwidth_gbps: Optional[float] = None  # Total bandwidth limit
    max_dsp: Optional[int] = None
    max_bram: Optional[int] = None
    max_uram: Optional[int] = None
    max_lut: Optional[int] = None
    
    def get_parallelism(self, kernel_name: str, interface_name: str) -> int:
        """Get parallelism for specific interface
        
        Args:
            kernel_name: Name of the kernel
            interface_name: Name of the interface
            
        Returns:
            Parallelism value for the interface
        """
        key = (kernel_name, interface_name)
        if key in self.interface_pars:
            return self.interface_pars[key]
        elif self.global_par is not None:
            return self.global_par
        else:
            return 1  # Default parallelism
    
    def apply_to_kernel(self, kernel: Kernel, kernel_name: str) -> Kernel:
        """Apply parallelism configuration to a kernel
        
        Updates stream dimensions based on parallelism settings.
        
        Args:
            kernel: Kernel to configure
            kernel_name: Instance name of the kernel in the graph
            
        Returns:
            New kernel with updated stream dimensions
        """
        new_interfaces = []
        
        for intf in kernel.interfaces:
            par = self.get_parallelism(kernel_name, intf.name)
            
            if par == 1:
                # No parallelism change
                new_interfaces.append(intf)
            else:
                # Compute new stream dimensions
                new_stream_dims = self._compute_stream_dims(intf, par)
                
                # Create new interface with updated stream dims
                new_intf = replace(intf, stream_dims=new_stream_dims)
                new_interfaces.append(new_intf)
        
        # Create new kernel with updated interfaces
        return replace(kernel, interfaces=new_interfaces)
    
    def _compute_stream_dims(self, interface: Interface, parallelism: int) -> Shape:
        """Compute stream dimensions for given parallelism
        
        Tries to factor parallelism to match tensor/block structure.
        
        Args:
            interface: Interface to parallelize
            parallelism: Target parallelism
            
        Returns:
            New stream dimensions
        """
        # Get current block dims (first phase for CSDF)
        if isinstance(interface.block_dims, list):
            block_dims = interface.block_dims[0]
        else:
            block_dims = interface.block_dims
        
        # Try to factor parallelism to match dimensions
        factors = self._factorize(parallelism)
        
        # Match factors to block dimensions
        stream_dims = []
        remaining_par = parallelism
        
        for i, bdim in enumerate(block_dims):
            # Find best factor for this dimension
            best_factor = 1
            for f in factors:
                if f <= bdim and bdim % f == 0 and remaining_par % f == 0:
                    best_factor = max(best_factor, f)
            
            stream_dims.append(best_factor)
            remaining_par //= best_factor
        
        # If we couldn't distribute all parallelism, put remainder in last dim
        if remaining_par > 1:
            stream_dims[-1] *= remaining_par
        
        return tuple(stream_dims)
    
    def _factorize(self, n: int) -> List[int]:
        """Get factors of n in ascending order"""
        factors = []
        for i in range(1, int(math.sqrt(n)) + 1):
            if n % i == 0:
                factors.append(i)
                if i != n // i:
                    factors.append(n // i)
        return sorted(factors)
    
    def validate_kernel(self, kernel: Kernel, kernel_name: str) -> Tuple[bool, Optional[str]]:
        """Validate if configuration satisfies kernel constraints
        
        Args:
            kernel: Kernel to validate
            kernel_name: Instance name
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Apply configuration
            configured = self.apply_to_kernel(kernel, kernel_name)
            
            # Validate kernel (checks pragmas)
            configured.validate()
            
            # Check stream dims don't exceed block dims
            for intf in configured.interfaces:
                if isinstance(intf.block_dims, list):
                    block_dims = intf.block_dims[0]
                else:
                    block_dims = intf.block_dims
                
                for i, (s, b) in enumerate(zip(intf.stream_dims, block_dims)):
                    if s > b:
                        return False, f"Stream dim {i} ({s}) exceeds block dim ({b}) for {intf.name}"
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    def estimate_resources(self, kernel: Kernel, kernel_name: str) -> Dict[str, float]:
        """Estimate resource usage for configured kernel
        
        Args:
            kernel: Kernel to estimate
            kernel_name: Instance name
            
        Returns:
            Dict of resource type to estimated usage
        """
        configured = self.apply_to_kernel(kernel, kernel_name)
        
        # Base resources from kernel
        resources = configured.resources.copy()
        
        # Scale by parallelism
        max_par = 1
        for intf in configured.interfaces:
            if intf.direction in [InterfaceDirection.INPUT, InterfaceDirection.OUTPUT]:
                max_par = max(max_par, prod(intf.stream_dims))
        
        # Simple scaling model
        for key in ["DSP", "LUT"]:
            if key in resources:
                resources[key] *= max_par
        
        # Bandwidth calculation
        bandwidth_reqs = configured.bandwidth_requirements()
        total_bandwidth_bits = sum(bandwidth_reqs.values())
        
        # Assume some clock frequency for bandwidth calculation
        clock_freq_mhz = 200  # Default 200 MHz
        total_bandwidth_bps = total_bandwidth_bits * clock_freq_mhz * 1e6
        
        resources["bandwidth_gbps"] = total_bandwidth_bps / 1e9
        
        return resources
    
    def __repr__(self) -> str:
        n_custom = len(self.interface_pars)
        global_str = f", global={self.global_par}" if self.global_par else ""
        return f"ParallelismConfig({n_custom} custom{global_str})"


@dataclass
class DSEConstraints:
    """Constraints for design space exploration
    
    Defines the requirements and limits that configurations must satisfy.
    """
    
    # Performance requirements
    min_throughput: Optional[float] = None  # inferences/sec
    max_latency: Optional[int] = None  # cycles
    min_fps: Optional[float] = None  # frames/sec
    
    # Resource limits (match ParallelismConfig resources)
    max_bandwidth_gbps: Optional[float] = None
    max_dsp: Optional[int] = None
    max_bram: Optional[int] = None
    max_uram: Optional[int] = None
    max_lut: Optional[int] = None
    max_power_w: Optional[float] = None
    
    # Parallelism bounds
    min_parallelism: int = 1
    max_parallelism: int = 64
    allowed_parallelisms: Optional[List[int]] = None  # If set, only these values
    
    # Frequency target
    target_frequency_mhz: float = 200.0
    
    def check_resources(self, resources: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Check if resource usage satisfies constraints
        
        Args:
            resources: Resource usage estimates
            
        Returns:
            Tuple of (all_satisfied, list_of_violations)
        """
        violations = []
        
        if self.max_bandwidth_gbps and resources.get("bandwidth_gbps", 0) > self.max_bandwidth_gbps:
            violations.append(f"Bandwidth {resources['bandwidth_gbps']:.1f} > {self.max_bandwidth_gbps} GB/s")
        
        if self.max_dsp and resources.get("DSP", 0) > self.max_dsp:
            violations.append(f"DSP {resources['DSP']} > {self.max_dsp}")
        
        if self.max_bram and resources.get("BRAM", 0) > self.max_bram:
            violations.append(f"BRAM {resources['BRAM']} > {self.max_bram}")
        
        if self.max_uram and resources.get("URAM", 0) > self.max_uram:
            violations.append(f"URAM {resources['URAM']} > {self.max_uram}")
        
        if self.max_lut and resources.get("LUT", 0) > self.max_lut:
            violations.append(f"LUT {resources['LUT']} > {self.max_lut}")
        
        if self.max_power_w and resources.get("power_w", 0) > self.max_power_w:
            violations.append(f"Power {resources['power_w']:.1f} > {self.max_power_w} W")
        
        return len(violations) == 0, violations
    
    def check_performance(self, metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Check if performance metrics satisfy constraints
        
        Args:
            metrics: Performance metrics
            
        Returns:
            Tuple of (all_satisfied, list_of_violations)
        """
        violations = []
        
        if self.min_throughput and metrics.get("throughput", 0) < self.min_throughput:
            violations.append(f"Throughput {metrics['throughput']:.1f} < {self.min_throughput} inf/s")
        
        if self.max_latency and metrics.get("latency", float('inf')) > self.max_latency:
            violations.append(f"Latency {metrics['latency']} > {self.max_latency} cycles")
        
        if self.min_fps and metrics.get("fps", 0) < self.min_fps:
            violations.append(f"FPS {metrics['fps']:.1f} < {self.min_fps}")
        
        return len(violations) == 0, violations
    
    def get_parallelism_range(self) -> List[int]:
        """Get valid parallelism values to explore
        
        Returns:
            List of parallelism values
        """
        if self.allowed_parallelisms:
            return [p for p in self.allowed_parallelisms 
                    if self.min_parallelism <= p <= self.max_parallelism]
        else:
            # Generate powers of 2 in range
            pars = []
            p = 1
            while p <= self.max_parallelism:
                if p >= self.min_parallelism:
                    pars.append(p)
                p *= 2
            return pars


@dataclass
class ConfigurationSpace:
    """Defines the space of configurations to explore
    
    Provides structured way to define which parallelism configurations
    should be explored for each interface.
    """
    
    # Interfaces to explore: (kernel_name, interface_name) -> list of parallelisms
    interface_options: Dict[Tuple[str, str], List[int]] = field(default_factory=dict)
    
    # Global options (if interface not in interface_options)
    global_options: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    
    # Coupling constraints: interfaces that must have same parallelism
    coupled_interfaces: List[List[Tuple[str, str]]] = field(default_factory=list)
    
    def add_interface(self, kernel_name: str, interface_name: str, 
                     options: Optional[List[int]] = None):
        """Add interface to exploration space
        
        Args:
            kernel_name: Kernel instance name
            interface_name: Interface name
            options: List of parallelism values to try
        """
        if options is None:
            options = self.global_options
        
        self.interface_options[(kernel_name, interface_name)] = options
    
    def add_coupling(self, interfaces: List[Tuple[str, str]]):
        """Add coupling constraint between interfaces
        
        Args:
            interfaces: List of (kernel_name, interface_name) that must have same parallelism
        """
        self.coupled_interfaces.append(interfaces)
    
    def generate_configs(self) -> List[ParallelismConfig]:
        """Generate all valid configurations in the space
        
        Returns:
            List of parallelism configurations
        """
        # Get all interfaces
        all_interfaces = list(self.interface_options.keys())
        
        # Group by coupling constraints
        groups = self._compute_coupling_groups(all_interfaces)
        
        # Generate configurations
        configs = []
        self._generate_recursive(groups, 0, {}, configs)
        
        return configs
    
    def _compute_coupling_groups(self, interfaces: List[Tuple[str, str]]) -> List[List[Tuple[str, str]]]:
        """Group interfaces by coupling constraints
        
        Args:
            interfaces: All interfaces
            
        Returns:
            List of groups, each group must have same parallelism
        """
        # Union-find to group coupled interfaces
        parent = {intf: intf for intf in interfaces}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Apply coupling constraints
        for group in self.coupled_interfaces:
            for i in range(1, len(group)):
                union(group[0], group[i])
        
        # Group by representative
        groups = {}
        for intf in interfaces:
            rep = find(intf)
            if rep not in groups:
                groups[rep] = []
            groups[rep].append(intf)
        
        return list(groups.values())
    
    def _generate_recursive(self, groups: List[List[Tuple[str, str]]], 
                           group_idx: int,
                           current_config: Dict[Tuple[str, str], int],
                           configs: List[ParallelismConfig]):
        """Recursively generate configurations
        
        Args:
            groups: Interface groups
            group_idx: Current group index
            current_config: Partial configuration being built
            configs: Output list
        """
        if group_idx >= len(groups):
            # Complete configuration
            configs.append(ParallelismConfig(interface_pars=current_config.copy()))
            return
        
        # Get options for this group
        group = groups[group_idx]
        
        # Use options from first interface in group
        options = self.interface_options.get(group[0], self.global_options)
        
        # Try each option
        for par in options:
            # Set parallelism for all interfaces in group
            for intf in group:
                current_config[intf] = par
            
            # Recurse to next group
            self._generate_recursive(groups, group_idx + 1, current_config, configs)
            
            # Backtrack
            for intf in group:
                del current_config[intf]