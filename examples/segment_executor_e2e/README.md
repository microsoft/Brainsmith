# Segment Executor End-to-End Test

This directory contains a complete end-to-end test demonstrating the segment executor architecture.

## Files

- `definitive_e2e_test.py` - The main test showing the complete flow
- `true_end_to_end_test.py` - Alternative test with detailed debugging output  
- `create_test_model.py` - Creates a simple ONNX model for testing
- `finn_steps_blueprint.yaml` - Blueprint with real FINN steps and branching
- `test_model.onnx` - Generated test model
- `END_TO_END_TEST_SUMMARY.md` - Detailed summary of what the test demonstrates
- `definitive_e2e/` - Output directory from test runs

## Running the Test

From the project root:

```bash
./smithy exec python examples/segment_executor_e2e/definitive_e2e_test.py
```

## What It Demonstrates

1. **Blueprint Parsing** - Parses YAML blueprints into design spaces
2. **Tree Building** - Builds execution trees with segments at branch points
3. **Segment Execution** - Executes segments using FINN's build_dataflow_cfg
4. **Transform Wrapping** - Multiple transforms wrapped as single FINN steps
5. **Artifact Sharing** - Models shared between parent and child segments
6. **Caching** - Detects cached segments based on output files

## Key Features Shown

- Mutually exclusive transform options (e.g., GiveUniqueNodeNames vs GiveRandomTensorNames)
- Optional transform stages (FoldConstants or skip)
- Multiple transforms per stage
- Hierarchical segment naming
- Proper error handling and reporting

See `END_TO_END_TEST_SUMMARY.md` for complete details.