# Folding Configuration Generator Analysis

## Overview
The `gen_initial_folding.py` script generates pre-computed folding configurations to avoid long RTL simulations for FIFO sizing. It creates specific PE (Processing Element) and SIMD (Single Instruction Multiple Data) parameters for each layer type.

## Key Parameters

### MVAU (Matrix Vector Activation Unit)
- **Standard layers (0-3)**: SIMD=`args.simd`, PE=`args.pe`
- **Layers 4-5**: SIMD=`2*args.simd`, PE=`2*args.pe` (doubled for larger layers)
- Runtime writeable weights: configurable
- Memory mode: `internal_decoupled`

### Thresholding
- PE=`args.other` (default: 4)
- Runtime writeable: 0 (fixed)
- **Critical**: This PE value determines output dimensions!

### DynMVU (Dynamic Matrix Vector Unit)
- PE=`args.pe`
- SIMD calculated based on divisibility:
  - If `args.simd % 3 == 0`: SIMD = `args.simd/3`
  - If `args.simd % 4 == 0`: SIMD = `args.simd/4`
  - Otherwise: SIMD = `args.simd`

### Other Components
- **DuplicateStreams**: PE=`args.other` (default: 4)
- **Shuffle**: SIMD=`args.other` (default: 4)
- **ElementwiseAdd/Mul**: PE=`args.other` (default: 4)
- **Softmax**: SIMD=`args.other` (default: 4)
- **LayerNorm**: SIMD=`args.other` (default: 4)

## Default Values
```bash
--simd 48    # MVAU SIMD
--pe 32      # MVAU PE
--other 4    # PE/SIMD for other operators
```

## Potential Issues

### 1. Thresholding PE Value
The script sets Thresholding PE to `args.other` (default 4), but our error trace shows some Thresholding nodes getting PE=1, which causes zero-dimension outputs.

### 2. Fixed Pattern Assumption
The script assumes a fixed pattern of nodes per layer:
- 6 MVAUs per layer
- 9 Thresholding nodes per layer
- 2 DynMVUs per layer
etc.

This might not match the actual model structure after transformations.

### 3. No Validation
The script doesn't validate that:
- PE values create compatible shapes between connected nodes
- SIMD/PE combinations are valid for the model dimensions
- The folding preserves data flow integrity

## Recommendations

### For quicktest.sh Usage
The script should be called with parameters that ensure:
1. **Thresholding PE > 1** to avoid zero dimensions
2. **Compatible SIMD/PE** values that match model dimensions
3. **Conservative values** for initial testing

### Example Safe Configuration
```bash
python gen_initial_folding.py \
    --simd 12 \
    --pe 8 \
    --other 1 \
    --num_layers 1 \
    --output safe_folding.json
```

But we need to ensure `--other` is set appropriately to avoid PE=1 issues.

### Better Approach
Instead of using this generic script, we should:
1. Generate folding based on actual model structure
2. Validate shape compatibility
3. Use FINN's auto-folding with constraints