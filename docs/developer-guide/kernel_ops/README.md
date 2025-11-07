# Kernel Op Developer Guide

A comprehensive guide to building FPGA hardware kernels using Brainsmith's kernel op system.

## What This Guide Covers

This guide teaches you how to:

- Define hardware operations declaratively using schemas
- Explore design spaces systematically
- Optimize configurations for latency, area, and power
- Handle complex patterns like broadcasting and multi-dimensional parallelization
- Write robust, maintainable kernel implementations

## Who Should Read This

This guide is for developers who:

- **Implement new hardware kernels** for neural network operations
- **Optimize existing kernels** through design space exploration
- **Integrate with FINN** dataflow compilation pipeline
- **Research hardware accelerators** and need systematic DSE

## Prerequisites

You should be familiar with:

- Python programming
- ONNX neural network format
- Basic FPGA/hardware concepts (parallelism, pipelining, resource constraints)
- Neural network operations (convolution, matrix multiply, etc.)

## Guide Structure

### [Chapter 1: Introduction](./01_introduction.md)

**What you'll learn:**
- Why the kernel op system exists
- Core philosophy and design principles
- When to use this system vs alternatives
- High-level architecture overview

**Read this if:** You're new to the system and want to understand the big picture.

### [Chapter 2: Fundamental Concepts](./02_fundamental_concepts.md)

**What you'll learn:**
- Three-level shape hierarchy (tensor → block → stream)
- Folding and hardware tradeoffs
- Design space vs design point
- Parallelization as divisor factorization
- Structural vs optimization parameters
- Immutable functional navigation

**Read this if:** You need the mental models to work effectively with the system.

### [Chapter 3: Building Your First Kernel](./03_building_your_first_kernel.md)

**What you'll learn:**
- Step-by-step kernel implementation
- Schema definition with real examples
- Testing and validation
- Exploring configurations
- Common patterns and troubleshooting

**Read this if:** You're ready to build a kernel and need a concrete example.

### [Chapter 4: Schema Design](./04_schema_design.md)

**What you'll learn:**
- Template system deep dive (FULL_SHAPE, FULL_DIM, tuples, callables)
- Datatype derivation strategies
- Constraint patterns and validation
- DSE dimension specification
- Layout requirements
- Complete schema examples

**Read this if:** You need to design complex kernel schemas with advanced features.

### [Chapter 5: Design Space Exploration](./05_design_space_exploration.md)

**What you'll learn:**
- Exploration strategies (exhaustive, greedy, Pareto)
- Navigation methods and patterns
- Evaluation and estimation
- Filtering and constraints
- Visualization techniques
- Multi-objective optimization

**Read this if:** You need to systematically optimize kernel configurations.

### [Chapter 6: Advanced Topics](./06_advanced_topics.md)

**What you'll learn:**
- Broadcasting support for elementwise operations
- Static vs dynamic tensor optimization
- Custom dimension derivation
- Multi-dimensional parallelization
- Custom DSE dimensions
- Internal datatypes
- Variable rank operations

**Read this if:** You're implementing complex kernels with advanced requirements.

### [Chapter 7: Best Practices](./07_best_practices.md)

**What you'll learn:**
- Schema design patterns
- Common pitfalls and solutions
- Testing strategies
- Performance optimization
- Debugging techniques
- Documentation guidelines
- Migration from legacy FINN

**Read this if:** You want to write robust, maintainable implementations.

## Quick Start

If you're impatient and want to dive in:

1. **Read the Introduction** to understand why this system exists
2. **Skim the Fundamental Concepts** to get the key mental models
3. **Follow the Tutorial** in Chapter 3 to build your first kernel
4. **Reference other chapters** as needed for specific features

## Quick Example

See [Chapter 3: Building Your First Kernel](./03_building_your_first_kernel.md) for a complete ChannelwiseAdd implementation tutorial with step-by-step guidance.

## Common Use Cases

| Goal | Relevant Chapters | Key Topics |
|------|------------------|------------|
| Implement new kernel | 3 → 4 → 7 | Schema design, KernelOp class, testing |
| Optimize existing kernel | 2 → 5 | Design space exploration, Pareto optimization |
| Add broadcasting support | 4 → 6 → 7 | BroadcastInfo helper, constraint removal |
| Multi-objective DSE | 5 | Objective functions, Pareto frontier |

## Additional Resources

### Code Examples

- Full kernel implementations: `brainsmith/kernels/`
- Test examples: `tests/kernels/`
- DSE examples: `examples/`

### API Reference

- Schema API: `brainsmith/dataflow/schemas.py`
- Constraint API: `brainsmith/dataflow/constraints.py`
- Navigation API: `brainsmith/dataflow/dse_models.py`
- Helper functions: `brainsmith/dataflow/spec_helpers.py`

### Related Documentation

- FINN Integration Guide
- HLS Backend Development
- RTL Backend Development
- Testing Framework Guide

## Getting Help

### Common Issues

See [Chapter 7: Best Practices](./07_best_practices.md) for:
- Common pitfalls and solutions
- Debugging techniques
- Troubleshooting checklist

### Questions?

- Check the troubleshooting section in Chapter 7
- Review examples in `brainsmith/kernels/`
- Open an issue on GitHub

## Contributing

When adding new kernels:

1. Follow patterns from Chapter 4
2. Write tests (Chapter 7)
3. Document schema intent (Chapter 7)
4. Consider edge cases (Chapter 7)

## Version History

- **v1.0** (2025-01) - Initial release with complete developer guide

## License

Copyright (c) Microsoft Corporation. Licensed under the MIT License.

---

**Ready to start?** Begin with [Chapter 1: Introduction](./01_introduction.md)!
