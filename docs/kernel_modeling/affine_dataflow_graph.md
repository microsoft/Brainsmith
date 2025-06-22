### Context Primer — Key Theory Behind Our Kernel-Modeling & Scheduling Stack

*(Everything you need even if you never opened the original Affine-Dataflow-Graph paper.)*

---

## 1 What an **Affine Dataflow Graph (ADFG)** Is

| Term from paper  | Intuitive meaning                                                                                                                                    | In our Python objects                                                                           |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Actor**        | A black-box task that fires repeatedly, taking some inputs and pushing some outputs. Its worst-case execution time (WCET) and token rates are known. | **`Kernel`** (stores `latency_cycles = (L_wc, L_avg)` and gets its rates from Interface tiles). |
| **Edge**         | FIFO channel that holds *tokens* between two actors.                                                                                                 | Edge inside **`DataflowGraph`**; token = one **Block/Tile** produced by source interface.       |
| **Token rates**  | Numbers *p* (produced) / *q* (consumed) per actor firing.                                                                                            | Computed from each interface’s **`block_seq` → `n_tokens_seq()`**.                              |
| **CSDF / UCSDF** | Cyclo-static patterns like *(2,1,1,2,…)* that repeat after P phases; captures ragged borders without state explosion.                                | Our ragged `block_seq` automatically becomes a CSDF sequence.                                   |

---

## 2 Timing Glue — Affine Relation & Period Search

*For every edge we can write an **affine clock equation***:

```
d · Tprod  =  n · Tcons  +  φ                   (Eq. 4 in the paper) :contentReference[oaicite:0]{index=0}
```

* `Tprod , Tcons` = periods of producer / consumer kernels.
* `(n, d)` = reduced ratio of their average token rates.
* `φ`      = constant offset from initial delays or pipeline bubbles.

**Why we care:** given these equalities we can **size the FIFO once and for all** and still let each kernel run on its own period.

### SRTA Period Search

Deadline-Monotonic priorities + “Shortest Remaining Time Adjustment”
(SRTA) iterate to the *shortest global period* that keeps every kernel
schedulable. We run the same loop verbatim; only the WCET numbers change
when you tweak parallelism.&#x20;

---

## 3 Buffer Depth Formula

Depth must cover the worst burst of tokens that can accumulate when
producer is faster than consumer during some window *t*:

```
depth ≥ max_t ( Produced(t) – Consumed(t) )       (Eq. 6) :contentReference[oaicite:2]{index=2}
```

With CSDF phases the produced/consumed counters are sums of *piecewise
linear* segments → still solvable by a tiny ILP (one integer var per
edge). Our `csdf_support.py` feeds those bounds into PuLP/OR-Tools.

---

## 4 How We Map Theory to Hardware Reality

| ADFG concept       | Our FPGA-aware refinement                                                                                                                                     |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Token size         | **Block / Tile (`BDIM_k`)** — smallest chunk that makes one firing worthwhile.                                                                                |
| Actor period       | Derived from **Initiation Interval** `II_k` (cycles to ship one block) and the SRTA search.                                                                   |
| WCET               | `latency_cycles.L_wc` **plus** optional priming & flush cycles for pipeline depth.                                                                            |
| Rate phases        | Generated from *ragged* `block_seq` (edge tiles, variable sequences).                                                                                         |
| Static feasibility | Enforced by **Pragmas**:<br>• `TIE expr₁ expr₂` for equality between interface widths<br>• `CONSTR expr rel value` for single-port limits (alignment, burst). |

---

## 5 Extra Real-World Effects We Layer On Top

1. **Sparsity** – `skip_prob` on each block phase lets perf estimator
   report *average FPS* while scheduler still protects the worst case.
2. **Pipeline latency** – `priming_cycles` & `flush_cycles` add
   batch-size dependent latency:
   `T_total(B) = Σ priming + B·period_graph + Σ flush`.
3. **Resource & power hooks** – each interface knows its per-cycle bit
   width (`dtype_bits × ipar`) so later passes can add DDR/NOC‐contention
   or energy models.

---

## 6 Illustrative Mini-Example

| Interface | `BDIM` pattern | `SDIM` (design vars) | Resulting CSDF rate       |
| --------- | -------------- | -------------------- | ------------------------- |
| `vec`     | `(M,)`         | `(ip_v,)`            | consumes 1 block / firing |
| `mat`     | `(N, M)`       | `(ip_v, ip_m)`       | consumes 1 block / firing |
| `out`     | `(N,)`         | `(ip_m,)`            | produces 1 block / firing |

Pragmas:

```
TIE      mat[1]   vec
CONSTR   mat[0]  %  SIMD
CONSTR   vec     %  BURST
```

* Scheduler explores `(ip_v, ip_m)` values that satisfy those equalities.
* SRTA picks minimal periods; ILP says buffer between `mat`→`out`
  needs `depth = LCM(ip_m, …)` tokens.
* Perf estimator boosts FPS if 50 % of `mat` rows are sparse
  (`skip_prob=0.5`).

---

### Take-Away

You only need five equations from the paper (affine clocks, CSDF
bounds, SRTA feasibility) plus our Interface/Pragma layer to get **tight
throughput guarantees** and **realistic latency/BRAM/DSP numbers** for
FPGA AI accelerators—without reading 30 pages of scheduling theory.

