**Context Cheat-Sheet: Core Ideas Adopted from the Affine-Dataflow-Graph Paper**

| Topic                                 | What we took from the paper                                                                                    | How we use it in our kernel-modeling framework                                                                                          |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Actor ⇔ Kernel**                    | ADFG “actors” have fixed worst-case execution time (WCET) and cyclo-static production/consumption rates.       | Every **Kernel** object stores `(L_wc, L_avg)` and phase-wise rates derived from its Interface tiles.                                   |
| **Edge**                              | FIFO that holds *tokens*; its depth must cover the worst burst implied by producer / consumer clocks.          | Our edge buffer ILP uses the same cumulative-difference bound over one hyper-period of the kernel firing pattern.                       |
| **Cyclo-Static (CSDF / UCSDF)**       | A finite sequence of rate phases `{p₀, p₁, …}` per actor edge; keeps ILP small while supporting ragged tiles.  | Ragged `block_seq` on each Interface becomes a CSDF pattern automatically; scheduler feeds phase rates to the original ILP formulation. |
| **Affine relation (n, φ, d)**         | Relates producer & consumer clocks: `d·T_prod = n·T_cons + φ`. Used to derive tight buffer bounds.             | We compute `(n, φ, d)` from Interface `II_k` for each edge and pass the tuple unchanged into the ADFG buffer-depth equations.           |
| **SRTA Period Search**                | Finds the shortest common period that makes every actor schedulable under fixed priorities.                    | Our *srt\_scheduler.py* reproduces SRTA verbatim; only the WCET input changes when designers sweep parallelism variables.               |
| **Priority Assignment (DM-like)**     | Deadline-Monotonic is near-optimal for single-period graphs.                                                   | Default `priority = latency_cycles.wc` order unless user overrides in kernel metadata.                                                  |
| **Buffer-Sizing ILP**                 | Min-cost depth subject to affine cumulative bounds; variables ≤ number of edges.                               | We drop the same constraints into PuLP/OR-Tools; no structural change required.                                                         |
| **Throughput vs. Latency Separation** | Paper optimises steady-state FPS; start-up latency treated separately.                                         | We added `priming_cycles` & `flush_cycles` per kernel—keeping steady-state math intact while giving realistic batch-1 numbers.          |

### Our Added Layers (not in the paper but built *on top* of it)

* **Three-tier Interface sizes** – *Tensor → Block → Stream* (TDIM/BDIM/SDIM) refine paper’s generic “token”.
* **Pragmas (`TIE`, `CONSTR`)** – declarative constraints on interface widths so the SRTA/ILP search space contains only legal points.
* **Sparsity & probabilities** – Dual latency and `skip_prob` plug into the *performance estimator* **after** the ADFG schedule is proven safe.

> **Bottom line:** we preserved every mathematical core of ADFG (affine bounds, CSDF phases, SRTA, ILP) and wrapped it with FPGA-centric abstractions (Interfaces, tiles, pragmas) so designers can change parallelism or tiling without rewriting the underlying real-time theory.

