# Design Space Exploration

Evaluate multiple hardware configurations to find optimal designs.

## Overview

Brainsmith uses segment-based DSE to efficiently explore large design spaces by reusing computation across similar configurations.

## Main Entry Point

::: brainsmith.dse.explore_design_space

## Blueprint Parsing

::: brainsmith.dse.parse_blueprint

## Advanced API

For power users who need fine-grained control over tree construction and execution:

::: brainsmith.dse.build_tree

::: brainsmith.dse.execute_tree

::: brainsmith.dse.SegmentRunner

## Configuration

::: brainsmith.dse.DSEConfig

## Design Space Definition

::: brainsmith.dse.GlobalDesignSpace

## Result Types

::: brainsmith.dse.TreeExecutionResult

::: brainsmith.dse.SegmentResult

::: brainsmith.dse.SegmentStatus

::: brainsmith.dse.OutputType

::: brainsmith.dse.ExecutionError

## Tree Structures

::: brainsmith.dse.DSETree

::: brainsmith.dse.DSESegment

## See Also

- [Design Space Exploration Architecture](../developer-guide/2-core-systems/design-space-exploration.md) - Conceptual overview and key concepts
- [Blueprints Reference](../developer-guide/3-reference/blueprints.md) - How to define design spaces declaratively
