# BrainSmith Examples

This directory contains examples demonstrating various features of BrainSmith.

## Available Examples

### segment_executor_e2e/
Complete end-to-end test showing the segment executor architecture in action. Demonstrates:
- Blueprint parsing from YAML
- Execution tree building with segments
- FINN integration for hardware compilation
- Transform wrapping and caching

See [segment_executor_e2e/README.md](segment_executor_e2e/README.md) for details.

### transform_registry_summary.md
Documentation of the transform registry system and available transforms.

## Running Examples

All examples should be run from the project root using smithy:

```bash
./smithy exec python examples/<example_path>
```