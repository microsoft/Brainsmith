## Performance Analysis script

This directory contains a rough performance analysis script and example notebook for attempting an early balance of a ONNX model for a given DSP resource constraint.

### Requirements

This script makes use of `CVXPY` version `1.6.5` for the constraint solving and SCIP as the underlying solver.
`CVXPY` can be installed easily with pip.
```bash
python3 -m pip install cvxpy==1.6.5
```

However, SCIP is a little more challenging to set up. The easiest way is to use Conda, however, this might have license considerations if conda is installed automatically in the docker flow.

```bash
conda install -c conda-forge scip 
```

### Usage:

1. Pass the cleaned model into the Analysis tool.
```python
from perf_analysis import PerfAnalysis
m = PerfAnalysis("genrec_cleanedup.onnx")
```

2. Call the balance method, optionally specify the resource constraints.
```python
res = m.balance(max_total_resources=7594)
```
Resources in this case is refering to DSPs. 

The solver will then run for about ~30s and then produce a dictionary of how many DSP resources should be allocated to each MatMul on the V80 along with the cycle estimate for that MatMul.

```
res = {'MatMul_0': {'DSPs': 357, 'cycles': 156474},
 'MatMul_1': {'DSPs': 357, 'cycles': 156474},
 'MatMul_2': {'DSPs': 117, 'cycles': 477133},
 'MatMul_3': {'DSPs': 348, 'cycles': 320690},
 'MatMul_4': {'DSPs': 1112, 'cycles': 320659},
 'MatMul_5': {'DSPs': 357, 'cycles': 312355},
 'MatMul_6': {'DSPs': 1142, 'cycles': 312388},
 'MatMul_7': {'DSPs': 519, 'cycles': 107778},
 'MatMul_8': {'DSPs': 1639, 'cycles': 136331},
 'MatMul_9': {'DSPs': 1639, 'cycles': 136331},
 'MatMul_10': {'DSPs': 5, 'cycles': 579338}}
```

Note that this has just been used so far to get rough estimates. These results should not be considered optimal, and there are various tweaks and restructuring of the cost function that can be done to improve the results.
