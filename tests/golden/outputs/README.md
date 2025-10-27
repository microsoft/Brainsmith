# Golden FINN Outputs

This directory contains pre-computed FINN build artifacts used as golden references for regression testing.

## Purpose

Golden outputs are expensive FINN builds (hours to days) that are:
- **Not committed to git** (too large, hardware-specific)
- **Regenerated on-demand** when FINN internals change
- **Used for cache behavior tests** and regression detection

## Directory Structure

```
outputs/
├── README.md (this file)
├── minimal_model/
│   ├── output.onnx
│   ├── reports/
│   └── metadata.json
├── branching_pipeline/
│   ├── path_a/
│   └── path_b/
└── rtl_simulation/
    └── ...
```

## Regenerating Golden Outputs

### When to Regenerate

Regenerate golden outputs when:
1. FINN API changes (breaking changes)
2. DSE execution behavior changes
3. Test failures indicate stale golden data
4. New test scenarios require new golden fixtures

### How to Regenerate

#### Option 1: Automatic (pytest flag)
```bash
# Regenerate all golden outputs
pytest -m finn_build --regenerate-golden

# Regenerate specific test
pytest tests/integration/finn/test_cache_behavior.py --regenerate-golden
```

#### Option 2: Manual (run FINN directly)
```bash
# Run FINN build with known-good inputs
brainsmith explore \
  --model tests/golden/models/minimal_model.onnx \
  --blueprint tests/golden/blueprints/minimal.yaml \
  --output tests/golden/outputs/minimal_model/

# Capture metadata
echo '{"generated_at": "2025-10-26", "finn_version": "0.10"}' > \
  tests/golden/outputs/minimal_model/metadata.json
```

### What to Commit

**Commit to git:**
- `tests/golden/models/` - Small ONNX models (< 1MB)
- `tests/golden/blueprints/` - Blueprint YAML files
- `tests/golden/trees/` - Syrupy snapshots (text)
- This README.md

**Do NOT commit:**
- `tests/golden/outputs/` - Large FINN artifacts (regenerate locally)

## Validation

Golden outputs should be validated after regeneration:

```bash
# Validate ONNX files
python -c "import onnx; onnx.load('tests/golden/outputs/minimal_model/output.onnx')"

# Run regression tests
pytest tests/integration/finn/ -v
```

## Maintenance Strategy

### Archive Old Outputs
When FINN changes significantly:
```bash
# Archive old outputs
mv tests/golden/outputs tests/golden/outputs.archive.$(date +%Y%m%d)

# Regenerate from scratch
pytest -m finn_build --regenerate-golden
```

### Storage Management
Golden outputs can be large (GBs for RTL/bitfile tests):
- Keep only latest versions
- Archive to external storage for historical reference
- Document FINN version compatibility in metadata.json

## Troubleshooting

**Golden outputs missing?**
1. Check if outputs are gitignored (expected)
2. Regenerate using pytest flag
3. Check FINN installation

**Tests failing after regeneration?**
1. Verify FINN version matches expected
2. Check for breaking changes in FINN
3. Review test assertions for hardcoded values

**Outputs too large?**
1. Use minimal models for fast tier tests
2. Defer RTL/hardware to separate storage
3. Consider mock-free golden fixtures strategy

---

**Last Updated:** Phase 3 (Test Workspace Setup)
**Status:** Skeleton created, awaiting Phase 4 fixture implementation
