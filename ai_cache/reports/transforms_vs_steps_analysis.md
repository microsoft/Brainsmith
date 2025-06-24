# Transforms vs Steps Analysis

## Key Insight
The current system conflates transforms and steps:
- **Transform**: A single operation/transformation (e.g., `FoldConstants`, `Streamline`)
- **Step**: A collection of transforms packaged together (e.g., `streamlining_step` contains multiple streamlining transforms)

## Current System Issues

### 1. LegacyConversionLayer
- Takes 6-entrypoint config (which should contain transforms)
- Maps them to build steps using `legacy_preproc` and `legacy_postproc`
- This works because it's designed for the legacy interface that expects steps

### 2. FINNEvaluationBridge
- `_combination_to_entrypoint_config()` maps DSE combinations to 6 entrypoints
- Currently treats these as transform names, but they get fed to legacy conversion

## Correct Architecture

### 6-Entrypoint System (Modern)
- Each entrypoint receives a list of **transforms**
- FINN inserts these transforms at 6 specific points in its workflow:
  1. After model loading
  2. After quantization setup
  3. After hardware kernel mapping
  4. After kernel specialization
  5. After kernel-level optimizations
  6. After graph-level optimizations

### Legacy System
- Receives a list of **build steps**
- Each step is a pre-packaged collection of transforms
- Steps are executed sequentially

## Implementation Plan

### Phase 1: Clean 6-Entrypoint Implementation
1. Create proper transform mapping (not step mapping)
2. Implement direct 6-entrypoint execution without legacy conversion
3. Properly handle transform insertion at the 6 points

### Phase 2: Legacy Translation Layer
1. Create transform-to-step packing logic
2. Map collections of transforms to appropriate legacy steps
3. Handle ordering and dependencies

## Transform Examples

### Individual Transforms (for 6-entrypoint)
- `FoldConstants`
- `InferShapes`
- `RemoveUnusedNodes`
- `ConvertBipolarToXnor`
- `Streamline`

### Build Steps (for legacy)
- `streamlining_step` (contains: Streamline, MoveLinearPastFork, etc.)
- `cleanup_step` (contains: InferShapes, FoldConstants, etc.)
- `qonnx_to_finn_step` (contains: conversion transforms)