**Kernel-Modeling Framework – Comprehensive Design Specification (v 3.1)**
*(integrates ragged-tiling, sparsity, pipeline latency, and the new
`TIE / CONSTR` pragma system)*

---

## 1  Purpose & Scope

This layer provides the **abstraction, data model, and APIs** that feed
the Affine-Data-flow-Graph (ADFG) scheduler for FPGA AI accelerators.
It must be:

| Requirement                                                                               | Reason                                   |
| ----------------------------------------------------------------------------------------- | ---------------------------------------- |
| **Accurate enough** to choose tiling & parallelism that fit DSP/BRAM and hit FPS targets. | First-order design exploration.          |
| **Lightweight** (pure Python 3.10, ≤ NumPy + networkx + optional ILP).                    | Notebook-friendly & CI-friendly.         |
| **Extensible** via declarative constraints (PRAGMAs) and plug-in cost / latency hooks.    | Future operators, buses, power-analysis. |

---

## 2  Core Abstractions

| Concept                 | Symbol                           | Description                                                                         |
| ----------------------- | -------------------------------- | ----------------------------------------------------------------------------------- |
| **Tensor**              | `TDIM`                           | Full payload on an interface for one inference.                                     |
| **Block / Tile**        | `BDIM_k`                         | *k*-th chunk that triggers **one kernel firing** (cyclo-static sequence `k=0…P−1`). |
| **Stream slice**        | `SDIM`                           | Elements transferred **per cycle** on that interface (flattened → `ipar`).          |
| **Initiation interval** | `II_k = ceil(prod(BDIM_k)/ipar)` | Cycles to ship block `k`.                                                           |
| **Latencies**           | `(L_wc, L_avg)`                  | Worst-case & average cycles from *first in* to *last out*.                          |
| **Pipeline costs**      | `P_c`, `F_c`                     | Priming & flush cycles per kernel.                                                  |
| **Pragmas**             | `TIE`, `CONSTR`                  | Declarative constraints on interface parallelism.                                   |

---

## 3  Module Layout

```
fpga_adfg/
├─ core/
│  ├─ interface.py      # TDIM/BDIM/SDIM + ragged & sparsity
│  ├─ kernel.py         # kernel + latency + pragmas
│  ├─ pragma.py         # TIE / CONSTR parsing & evaluation
│  ├─ graph.py          # data-flow graph wrapper (networkx)
│  └─ rates.py          # affine & CSDF helpers
├─ schedule/
│  ├─ srt_scheduler.py  # DM + SRTA + ILP
│  └─ csdf_support.py   # phase-wise rate bounds
├─ analysis/
│  ├─ perf.py           # FPS estimator with sparsity & pragmas
│  └─ latency.py        # batch-aware latency calculator
├─ resources/
│  └─ cost_models.py    # pluggable resource power models
└─ tests/               # exhaustive unit tests
```

---

## 4  Data Classes (API)

### 4.1  Interface

```python
@dataclass
class Interface:
    name: str
    direction: Literal["in", "out"]
    dtype_bits: int
    tensor_shape: Shape                     # TDIM
    block_seq: List[Optional[Shape]] = ...  # ragged BDIMs (None = optional)
    stream_shape: Shape = (1,)              # SDIM
    skip_prob: List[float] = ...            # sparsity (∈[0,1])

    # derived ----------
    def ipar(self) -> int              # ∏ SDIM
    def ii_seq(self) -> List[int]      # II_k
    def n_tokens_seq(self) -> List[int]# blocks/inference per phase
```

### 4.2  Pragmas (`core/pragma.py`)

| Class          | Syntax                                                | Semantics                                                                    |
| -------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------- |
| `TiePragma`    | `TIE <expr₁> <expr₂>`                                 | Equality between two expressions that each reference ≥ 1 interface.          |
| `ConstrPragma` | `CONSTR <expr> <rel> <value>`<br>`rel ∈ {=, ≤, ≥, %}` | Unary constraint on one interface expression against constant/sample symbol. |

**Expression grammar** (mini-AST parsed with `ast.parse`):

```
<expr> ::= <factor> { ("+"|"-"|"*"|"/") <factor> }*
<factor> ::= <iface-name> ["[" axis "]"]
           | <integer-const> | <symbol> | "(" <expr> ")"
```

`kernel.pragma_env: dict` supplies compile-time symbols (`BURST`, `SIMD`,
`ALIGN`, …).

### 4.3  Kernel

```python
@dataclass
class Kernel:
    name: str
    interfaces: List[Interface]
    latency_cycles: Tuple[int,int]          # (L_wc, L_avg)
    priming_cycles: Optional[int] = None
    flush_cycles:   int = 0
    resources: dict = field(default_factory=dict)
    pragma_env: Dict[str,int] = field(default_factory=dict)
    pragmas: List[Pragma] = field(default_factory=list)

    # derived ----------
    def initiation_interval(self) -> int
    def firing_pattern(self) -> List[Dict[str,Shape]]
    def validate(self) -> None              # raises ConstraintError
    def to_adfg_rates(self) -> Dict[str, List[int]]
```

---

## 5  Constraint Handling Workflow

```
user kernel YAML/JSON
        │
        ▼
Kernel() ctor ──►  validate()
        │          ├─ Evaluate TIE equalities
        │          └─ Check CONSTR bounds / multiples
        └─ On error → ConstraintError with offending rule
```

The **design-space enumerator** (or ILP) requests candidate `ipar`
tuples; any tuple failing a pragma is discarded before scheduling.

---

## 6  Scheduler & Analysis Integration

| Stage                     | Uses                                       | Notes                                         |
| ------------------------- | ------------------------------------------ | --------------------------------------------- |
| **SRTA period search**    | `initiation_interval()`, worst-case `II_k` | CSDF phases handled in `csdf_support.py`.     |
| **ILP buffer sizing**     | `to_adfg_rates()`                          | Phase-wise cumulative production/consumption. |
| **Performance estimator** | `latency_cycles.avg`, `skip_prob`          | Closed-form expectation or Monte-Carlo.       |
| **Latency annotator**     | `priming_cycles`, `flush_cycles`           | Computes `T_total(batch)`.                    |

---

## 7  Typical Usage

```python
from fpga_adfg.core import Interface, Kernel
from fpga_adfg.pragma import TiePragma, ConstrPragma

vec = Interface("vec", "in", 16,
                tensor_shape=(1,M),
                block_seq=[(M,)],
                stream_shape=(ip_v,))

mat = Interface("mat", "in", 16,
                tensor_shape=(N,M),
                block_seq=[(N,M)],
                stream_shape=(ip_v, ip_m))

out = Interface("out", "out", 16,
                tensor_shape=(N,),
                block_seq=[(N,)],
                stream_shape=(ip_m,))

vxmat = Kernel(
    "vxmat",
    interfaces=[vec, mat, out],
    latency_cycles=(900,600),
    priming_cycles=300,
    pragma_env={"SIMD":16, "BURST":64},
    pragmas=[
        TiePragma("mat[1]", "vec"),           # axis tie
        ConstrPragma("mat[0]", "%", "SIMD"),  # multiple of SIMD
        ConstrPragma("vec", "%", "BURST")     # AXI burst alignment
    ]
)
vxmat.validate()
```

---

## 8  Testing Matrix

| ID   | Feature                    | Assertion                                            |
| ---- | -------------------------- | ---------------------------------------------------- |
| T-01 | Legacy kernel (no pragmas) | Validates & schedules identical to v 2.0.            |
| T-02 | Ragged tiling (edge tiles) | Buffer depth == analytical maximum diff.             |
| T-03 | Sparsity skip\_prob        | Expected FPS within 2 % of Monte-Carlo 10 k runs.    |
| T-04 | Pipeline latency           | `batch=1` latency matches sum(priming)+period+flush. |
| T-05 | Pragma equality            | Violating `TIE` raises `ConstraintError`.            |
| T-06 | Pragma unary multiple      | `%` relation respected across enumeration.           |

---

## 9  Migration Guide

* **Earlier enum‐style pragmas**: `Kernel.convert_old_pragmas()` rewrites
  to `TiePragma/ConstrPragma`.
* **Single-block kernels**: keep `block_seq=[]`, `stream_shape=(ipar,)`.
* **No sparsity**: leave `skip_prob=[]` (implicitly zeros).

---

## 10  Extensibility Hooks

| Hook                                   | Purpose                                       |
| -------------------------------------- | --------------------------------------------- |
| `Interface.energy_per_token`           | Add power modeling.                           |
| `Kernel.latency_fn(params)`            | Data-dependent latency (seq-length).          |
| Custom pragma symbols                  | E.g. `LANES`, `DDR_BEAT`, injected per board. |
| Extra `CONSTR` relations (`!=`, `gcd`) | Add via one enum line & AST op.               |

---

## 11  Road-map

1. **v 3.1 release** – ship this spec; CI green.
2. **v 3.2** – shared-bus contention model (resource-class tags).
3. **v 3.3** – hierarchical (DRAM→SRAM→REG) multi-level interfaces.
4. **v 3.4** – power & thermal estimation, clock-domain pragmas.

---

### 12  Summary

This document defines a **feature-complete, declarative kernel model**
featuring:

* Multi-phase (ragged) tiling
* Probabilistic sparsity handling
* Pipeline start-up / flush latency
* Two concise yet expressive pragma types (`TIE`, `CONSTR`)

Together, these give architects high-confidence throughput *and*
single-batch latency predictions while keeping the code base
maintainable and extensible for future FPGA and AI operator innovations.

