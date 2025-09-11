# Performance Optimizations Summary

## CPU Threading Optimizations

### 1. Main Calculations (QSpin/Cluster Model)
- **Location**: `aah_code/cluster_model/runpod_runs/cpu_optimization.py`
- **Activation**: Automatic via `optimize_cpus()` in main.py
- **Effect**: Uses 48 physical cores (out of 96 vCPUs)
- **Performance gain**: ~11x speedup on matrix operations

### 2. DMRG Calculations
- **Location**: `aah_code/real_space_dmrg.py`
- **Activation**: Automatic on import
- **Effect**: Sets OMP_NUM_THREADS=48 for TenPy/NumPy/OpenBLAS
- **Note**: DMRG has inherent sequential parts that limit parallelization

### 3. Environment Variables Set
```bash
OMP_NUM_THREADS=48
OPENBLAS_NUM_THREADS=48
MKL_NUM_THREADS=48
VECLIB_MAXIMUM_THREADS=48
NUMEXPR_NUM_THREADS=48
```

## Verbose Output Control

### Clean Progress Bars
- Progress bars now stay visible without being overwritten
- Verbose debug prints suppressed by default
- QSpin hermiticity/symmetry check messages suppressed

### Enabling Verbose Mode
```bash
# From command line:
CLUSTER_VERBOSE=true python main.py  # Info messages
CLUSTER_DEBUG=true python main.py    # Debug messages

# Or in Python:
from aah_code.cluster_model.logging_config import set_verbose
set_verbose(True)
```

## DMRG Performance Notes

DMRG calculations can still be slow because:
1. **Algorithm limitations**: DMRG sweeps are inherently sequential
2. **Bond dimension**: Higher chi values increase computation time exponentially
3. **System size**: Scales roughly as O(L × chi³)

### Recommendations for Faster DMRG:
- Use smaller `chi` values for initial tests (16-32)
- Use finite DMRG only when necessary (infinite is faster for uniform systems)
- Consider reducing convergence criteria for exploratory runs

## File Download Optimization

### RunPod to Local Transfer
```bash
# Configured in: aah_code/cluster_model/runpod_runs/config.yaml
scp -P 22131 root@69.30.85.116:/path/to/file ~/Desktop/runpod_results/
```

Auto-sync runs after calculations to create combined HTML files.