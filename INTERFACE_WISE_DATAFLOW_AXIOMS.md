# Interface-Wise Dataflow Modeling: Core Axioms

## 1. Data Hierarchy
```
Tensor → Block → Stream → Element
```
- **Tensor**: Complete data (entire hidden state/weight)
- **Block**: Minimum data for one calculation
- **Stream**: Data per clock cycle
- **Element**: Single value

## 2. The Core Relationship
```
tensor_dims → chunked into → num_blocks pieces of shape block_dims → streamed as stream_dims per cycle
```
- **tensor_dims**: Full tensor shape (no batch dimension)
- **num_blocks**: Number of blocks available
- **block_dims**: Shape of each block
- **stream_dims**: Data streamed per clock cycle

## 3. Interface Types
- **Input**: AXI-Stream activation data in
- **Output**: AXI-Stream activation data out  
- **Weight**: AXI-Stream weight data in
- **Config/Control**: AXI-Lite (excluded from dataflow model)

## 4. Computational Model
Each **Input block** + **Weight block** → **Output block**
- **cII**: Cycles per calculation (Input block × Weight block)
- **eII**: Cycles per execution (Input block × Weight tensor)
- **L**: Cycles per inference (Input tensor)

## 5. Parallelism Parameters
- **iPar**: Input parallelism (SIMD), $1 ≤ iPar ≤ block_dims_I$
- **wPar**: Weight parallelism (PE), $1 ≤ wPar ≤ tensor_dims_W$

## 6. Stream Relationships
- $stream\_dims_I = iPar$
- $stream\_dims_W = wPar × iPar × (block\_dims_W / block\_dims_I)$
- $stream\_dims_O = stream\_dims_I × (block\_dims_O / block\_dims_I)$

## 7. Timing Relationships  
- $cII = ∏(block\_dims_I / stream\_dims_I)$
- $eII = cII × ∏(tensor\_dims_W / wPar)$
- $L = eII × ∏(tensor\_dims_I)$

## 8. Tiling Constraint
Each level tiles into the next: stream → block → tensor

## 9. Layout-Driven Chunking
ONNX tensor layout determines chunking dimension:
- `[N, C, H, W]` → chunk along C
- `[N, L, C]` → chunk along L  
- `[N, H, W, C]` → chunk along H×W

## 10. Runtime Extraction
All parameters (tensor_dims, block_dims, stream_dims) determined at runtime from ONNX pattern and DSE results.