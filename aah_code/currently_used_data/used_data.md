## Figure Status

Source of truth: [`figures.csv`](figures.csv) — edit that file directly.

### Fig 2

| Paper Ref | Description | Data | PDF | Status | Data func | Plot func | Notes |
|-----------|-------------|------|-----|--------|-----------|-----------|-------|
| Fig 2 | Relative error vs U for V=0 (pure Hubbard) comparing in | Y | Y | DONE | compare_filling_with_int_cluster_sizes | plot_hub_comparison | use_bethe=True for exact Bethe ansatz reference instead of DMRG |
| App Fig 2a | Absolute energy vs U for V=0 comparing interaction sepa | Y | - | NEEDS PLOT | compare_filling_with_int_cluster_sizes | - | Same data as Fig 2 main, replot with plot_relative_error=False |
| App Fig 2b | Filling relative error vs U for V=0 comparing interacti | Y | - | NEEDS PLOT | compare_compressibility_with_int_cluster_sizes | - | Rel error of filling (cluster vs DMRG) |
| App Fig 2c | Filling nu vs mu_0 for V=0 comparing interaction separa | Y | Y | NEEDS REVIEW | compare_compressibility_with_int_cluster_sizes | plot_hub_compressibility | PDF generated but needs visual review |
| App Fig 2d | Absolute energy vs U for V=0 zoomed to U in [0 1] with  | Y | Y | NEEDS REVIEW | compare_filling_with_int_cluster_sizes | plot_hub_smallU_abs_energy |  |
| App Fig 2e | Filling nu vs mu_0 for V=0 zoomed to U in [0 1] with 10 | Y | Y | NEEDS REVIEW | compare_compressibility_with_int_cluster_sizes | plot_hub_smallU_compressibility |  |

### Fig 4

| Paper Ref | Description | Data | PDF | Status | Data func | Plot func | Notes |
|-----------|-------------|------|-----|--------|-----------|-----------|-------|
| Fig 4 (low U) | Relative error vs V for maximal separation beta=1/2 low | Y | Y | DONE | compare_filling_cluster_sizes | plot_v_convergence('1-2', U_subset=[0,1,2,3]) |  |
| Fig 4 (high U) | Relative error vs V for maximal separation beta=1/2 hig | Y | Y | DONE | compare_filling_cluster_sizes | plot_v_convergence('1-2', U_subset=[5,7,10,20], suffix='_highU') |  |
| Fig 4 appendix (beta=1/3) | Relative error vs V for maximal separation beta=1/3 | Y | Y | DONE | compare_filling_cluster_sizes | plot_v_convergence('1-3', U_subset=[0,1,3,5]) |  |
| Fig 4 appendix (beta=1/4) | Relative error vs V for maximal separation beta=1/4 | Y | Y | DONE | compare_filling_cluster_sizes | plot_v_convergence('1-4', U_subset=[0,1,3,5]) | Uses L=40 (not L=48) |
| App Fig 4 | Filling nu vs mu_0 for finite V comparing cluster sizes | Y | Y | NEEDS REVIEW | compare_compressibility_cluster_sizes | plot_v_compressibility | PDF generated but needs visual review |

### Fig 5

| Paper Ref | Description | Data | PDF | Status | Data func | Plot func | Notes |
|-----------|-------------|------|-----|--------|-----------|-----------|-------|
| Fig 5 | Maximal vs non-maximal clustering at fixed supercluster | Y | Y | DONE | compare_fixed_supercluster | plot_fixed_supercluster |  |
| App Fig 5 | Filling nu vs mu_0 at fixed supercluster size (compress | - | - | NEEDS DATA | - | - | No data or plot code yet. Needs computation function + plotting function. |


### PBC vs OBC Convergence Benchmarking

| Item | Description | Location |
|------|-------------|----------|
| Fig 4 chi convergence plot | PBC vs OBC energy per site vs chi for U={0,2}, V={0,1,5}, L=48, half-fill, v_sep=(1,2) | `sandbox/pbc_fig4_convergence.png` |
| Fig 4 chi convergence data | Pickle with all energies and timings | `sandbox/pbc_fig4_convergence_results.pkl` |
| Fig 4 chi convergence script | Benchmark script (incremental save) | `sandbox/pbc_fig4_convergence.py` |
| Bethe ansatz convergence plot | PBC vs OBC vs Bethe ansatz for V=0 Hubbard, L={16,32,48} | `sandbox/pbc_convergence_combined.png` |
| Bethe ansatz convergence data | Pickle with all energies and timings | `sandbox/pbc_convergence_results.pkl` |
| Bethe ansatz convergence script | Benchmark script | `sandbox/pbc_convergence.py` |
| PBC finite-size analysis writeup | Full writeup of PBC vs OBC finite-size scaling | `sandbox/pbc_finite_size_analysis.md` |
| PBC DMRG implementation | `get_gnd_pbc()`, `get_gnd_fixed_filling_pbc()`, `RealSpaceHubbard1D_PBC` | `aah_code/real_space_dmrg.py` |
| BC dispatch | `get_gnd(..., bc='periodic')` and `get_gnd_fixed_filling(..., bc='periodic')` dispatch to PBC variants | `aah_code/real_space_dmrg.py` |
| Fig 2 pipeline PBC flag | `compare_filling_with_int_cluster_sizes(..., finite_dmrg_bc='periodic')` | `aah_code/cluster_model/plots.py` |
| CLI PBC flag | `--finite_dmrg_bc periodic` | `aah_code/cluster_model/cluster_runner.py` |

## Programmatic access

```python
from aah_code.currently_used_data.figure_registry import load_figures, Status
figures = load_figures()
todo = {k: v for k, v in figures.items() if v['status'] != Status.DONE}
```

## Key locations

- **Run functions**: `aah_code/cluster_model/plots.py`
- **CLI wrappers**: `aah_code/cluster_model/cluster_runner.py`
- **Merge functions**: `aah_code/cluster_model/merge_comparison_results.py`
- **Plot functions**: `aah_code/cluster_model/large_files/paper_plots/make_paper_plots.py`
- **Data pickles**: `aah_code/cluster_model/large_files/plots/` and `large_files/partials/`
- **Output PDFs**: `aah_code/cluster_model/large_files/paper_plots/`
