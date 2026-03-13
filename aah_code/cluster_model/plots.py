"""
Plotting functions for comparing different cluster model setups with iDMRG.
"""

import numpy as np
import time
import csv
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Any, Tuple, List, Dict, Optional, Sequence, Union
import os
import warnings
import pickle
from datetime import datetime
from pathlib import Path

from aah_code.cluster_model.model import ClusterModelConfig, PhysicalParams
from aah_code.cluster_model.clustering import generate_clusters
from aah_code.cluster_model.run_scripts_me import (
    get_general_expectations,
    get_general_density_wave_observable,
    get_general_spectra,
)
from aah_code.cluster_model.gpu_mu_sweep import batched_mu_expectations
from aah_code.real_space_dmrg import (
    run_dmrg_method,
    get_gnd,
    get_gnd_fixed_filling,
    get_finite_dmrg_density_wave_observable,
)

# Configure plotly to work outside of notebooks
pio.renderers.default = "browser"

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable


class TimingRecorder:
    """Collects timing records when enabled."""
    def __init__(
        self,
        *,
        csv_path: Optional[Union[str, os.PathLike]] = None,
        csv_only_method: Optional[str] = "cluster_ED_supercluster",
    ):
        self.records: List[Dict] = []
        self.csv_path = Path(csv_path).expanduser().resolve() if csv_path else None
        self.csv_only_method = csv_only_method
        self._csv_fh = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_fieldnames = [
            "timestamp",
            "method",
            "elapsed_sec",
            "U",
            "V",
            "t",
            "L",
            "Nc",
            "supercluster_index",
            "super_cluster_size",
            "int_sep",
            "v_sep",
            "slurm_job_id",
            "slurm_array_job_id",
            "slurm_array_task_id",
            "pid",
            "meta_json",
        ]
        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_fh = self.csv_path.open("a", newline="")
            self._csv_writer = csv.DictWriter(self._csv_fh, fieldnames=self._csv_fieldnames)
            if self.csv_path.stat().st_size == 0:
                self._csv_writer.writeheader()
                self._csv_fh.flush()

    def record(self, **kwargs):
        self.records.append(kwargs)
        if self._csv_writer is None or self._csv_fh is None:
            return

        method = kwargs.get("method")
        if self.csv_only_method is not None and method != self.csv_only_method:
            return

        def fmt_ratio(value) -> str:
            if value is None:
                return ""
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return f"{value[0]},{value[1]}"
            return str(value)

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "method": "" if method is None else str(method),
            "elapsed_sec": kwargs.get("elapsed_sec", ""),
            "U": kwargs.get("U", ""),
            "V": kwargs.get("V", ""),
            "t": kwargs.get("t", ""),
            "L": kwargs.get("L", ""),
            "Nc": kwargs.get("Nc", ""),
            "supercluster_index": kwargs.get("supercluster_index", ""),
            "super_cluster_size": kwargs.get("super_cluster_size", ""),
            "int_sep": fmt_ratio(kwargs.get("int_sep")),
            "v_sep": fmt_ratio(kwargs.get("v_sep")),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
            "pid": os.getpid(),
            "meta_json": json.dumps(kwargs, default=str, sort_keys=True),
        }
        self._csv_writer.writerow(row)
        self._csv_fh.flush()


def time_call(recorder: Optional[TimingRecorder], meta: Dict, func, *args, **kwargs):
    """Run func(*args, **kwargs), recording elapsed time into recorder if provided."""
    if recorder is None:
        return func(*args, **kwargs)
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    meta = dict(meta)
    meta["elapsed_sec"] = elapsed
    recorder.record(**meta)
    return result


def supercluster_size_from_clusters(clusters: np.ndarray) -> Optional[int]:
    """
    Infer total supercluster size from a clusters array.

    Expected shapes often look like [num_super, n, Nc, ...], so we take the
    product of the second and third dimensions when available. If deeper dims
    are present, include them; if only [num_super, n], fall back to n.
    """
    try:
        if clusters.ndim >= 3:
            return int(np.prod(clusters.shape[1:3]))
        elif clusters.ndim >= 2:
            return int(clusters.shape[1])
    except Exception:
        return None
    return None


def plot_timings(
    timings: List[Dict],
    output_dir: str = 'large_files/plots',
    filename_prefix: str = 'timing_summary',
    show_plots: bool = True,
):
    """
    Plot average runtime per supercluster size (or Nc fallback) with min/max error bars.
    """
    if not timings:
        raise ValueError("No timing records provided.")

    # Aggregate by method and supercluster size (fallback to Nc)
    agg: Dict[Tuple[str, float], List[float]] = {}
    for rec in timings:
        method = rec.get("method", "unknown")
        sc_size = rec.get("super_cluster_size")
        if sc_size is None:
            sc_size = rec.get("Nc", np.nan)
        try:
            sc_size = float(sc_size)
        except Exception:
            sc_size = np.nan
        key = (method, sc_size)
        agg.setdefault(key, []).append(float(rec.get("elapsed_sec", np.nan)))

    method_groups: Dict[str, Dict[str, List[float]]] = {}
    for (method, sc_size), vals in agg.items():
        vals = [v for v in vals if np.isfinite(v)]
        if not vals or not np.isfinite(sc_size):
            continue
        method_groups.setdefault(method, {"x": [], "mean": [], "err_up": [], "err_down": []})
        mean = float(np.mean(vals))
        vmax = float(np.max(vals))
        vmin = float(np.min(vals))
        method_groups[method]["x"].append(sc_size)
        method_groups[method]["mean"].append(mean)
        method_groups[method]["err_up"].append(vmax - mean)
        method_groups[method]["err_down"].append(mean - vmin)

    fig = go.Figure()
    for method, data in method_groups.items():
        # sort by x
        order = np.argsort(data["x"])
        x = [data["x"][i] for i in order]
        y = [data["mean"][i] for i in order]
        err_up = [data["err_up"][i] for i in order]
        err_down = [data["err_down"][i] for i in order]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                name=method,
                error_y=dict(
                    type='data',
                    array=err_up,
                    arrayminus=err_down,
                    visible=True,
                ),
                hovertemplate="Supercluster=%{x}<br>mean=%{y:.4f}s<br>+%{error_y.array:.4f}/-%{error_y.arrayminus:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Runtime per supercluster size",
        xaxis_title="Supercluster size (sites)",
        yaxis_title="Elapsed time (s)",
        hovermode="x unified",
    )

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{filename_prefix}_{timestamp}.html"
    filepath = os.path.join(output_dir, filename)
    fig.write_html(filepath)

    if show_plots:
        fig.show()

    return fig, {"html": filepath}


def format_sep_as_pi(sep_tuple: Tuple[int, int]) -> str:
    """Convert separation ratio to π fraction notation."""
    numerator = 2 * sep_tuple[0]
    denominator = sep_tuple[1]
    if denominator == 0:
        return "undefined"
    if numerator == denominator:
        return 'π'
    if numerator == 1:
        return f'π/{denominator}'
    from fractions import Fraction
    frac = Fraction(numerator, denominator)
    if frac.denominator == 1:
        if frac.numerator == 1:
            return 'π'
        return f'{frac.numerator}π'
    if frac.numerator == 1:
        return f'π/{frac.denominator}'
    return f'{frac.numerator}π/{frac.denominator}'


def compare_int_seps_with_dmrg(
    v_sep_ratio: Tuple[int, int],
    int_sep_list: List[Tuple[int, int]],
    U_values: np.ndarray = None,
    V_values: np.ndarray = None,
    t: float = None,
    x_axis: dict = None,
    varying_parameter: dict = None,
    fixed_parameter: dict = None,
    L: int = 20,
    Nc: int = 2,
    chi: int = 32,
    solver_method: str = 'dense_ED',
    states_retained: int = 4,
    output_dir: str = 'large_files/plots',
    show_plots: bool = True,
    save_pickle: bool = True,
    include_idmrg: bool = True,
    include_finite_dmrg: bool = False,
    include_timing: bool = False,
    include_timing_plot: bool = False,
):
    """
    Compare different int_sep setups with iDMRG for a fixed v_sep.
    
    Can be called in two ways:
    1. Legacy mode: Using U_values, V_values, and t directly
    2. Flexible mode: Using x_axis, varying_parameter, and fixed_parameter dicts
    
    Args:
        v_sep_ratio: Fixed V separation ratio (e.g., (1,2) for staggered)
        int_sep_list: List of int_sep ratios to compare (e.g., [(1,2), (1,3), (1,4)])
        U_values: (Legacy) Array of U values to sweep
        V_values: (Legacy) Array of V values to test
        t: (Legacy) Hopping parameter
        x_axis: Dict with single key-value pair for x-axis parameter (e.g., {'U': np.array([...])})
        varying_parameter: Dict with single key-value pair for subplot parameter (e.g., {'V': np.array([...])})
        fixed_parameter: Dict with single key-value pair for fixed parameter (e.g., {'t': 1.0})
        L: System size
        Nc: Cluster size
        chi: DMRG bond dimension
        solver_method: Method for solving ('dense_ED' or 'sparse_ED')
        states_retained: Number of states retained in solver
        output_dir: Directory to save plots
        show_plots: Whether to display plots
        save_pickle: Whether to save results to pickle file (default: True)
        include_idmrg: Whether to include infinite DMRG (iDMRG) calculations (default: True). 
                      Set to False to only compare cluster methods.
        include_finite_dmrg: Whether to also include finite DMRG calculations (default: False).
                            This adds finite-size DMRG results alongside iDMRG.
    
    Returns:
        figures: List of plotly figures
        all_results: Dictionary with all computed results
    """
    timing_recorder = TimingRecorder() if include_timing else None
    
    # Handle parameter input modes
    if x_axis is not None and varying_parameter is not None and fixed_parameter is not None:
        # New flexible mode
        x_param_name, x_values = next(iter(x_axis.items()))
        varying_param_name, varying_values = next(iter(varying_parameter.items()))
        fixed_param_name, fixed_value = next(iter(fixed_parameter.items()))
        
        # Convert to numpy arrays if they're lists
        x_values = np.array(x_values) if isinstance(x_values, list) else x_values
        varying_values = np.array(varying_values) if isinstance(varying_values, list) else varying_values
        
        # Create parameter mapping
        param_names = {'x': x_param_name, 'varying': varying_param_name, 'fixed': fixed_param_name}
    else:
        # Legacy mode
        if U_values is None or V_values is None or t is None:
            raise ValueError("Must provide either (U_values, V_values, t) or (x_axis, varying_parameter, fixed_parameter)")
        x_param_name, x_values = 'U', U_values
        varying_param_name, varying_values = 'V', V_values
        fixed_param_name, fixed_value = 't', t
        param_names = {'x': 'U', 'varying': 'V', 'fixed': 't'}
    
    print("=" * 60)
    if include_idmrg or include_finite_dmrg:
        dmrg_types = []
        if include_idmrg:
            dmrg_types.append("iDMRG")
        if include_finite_dmrg:
            dmrg_types.append("finite DMRG")
        print(f"Comparing {len(int_sep_list)} int_sep configurations with {' and '.join(dmrg_types)}")
    else:
        print(f"Comparing {len(int_sep_list)} int_sep configurations (no DMRG)")
    print("=" * 60)
    print(f"System: L={L}, Nc={Nc}")
    print(f"Fixed parameter: {fixed_param_name}={fixed_value}")
    print(f"Fixed v_sep={v_sep_ratio}")
    print(f"Int_sep configurations: {int_sep_list}")
    if include_idmrg or include_finite_dmrg:
        print(f"DMRG: chi={chi}")
    print(f"X-axis ({x_param_name}): {x_values}")
    print(f"Varying parameter ({varying_param_name}): {varying_values}")
    
    # Storage for all results and failed calculations
    all_results = {}
    failed_calculations = []
    
    for vary_val in tqdm(varying_values, desc=f"{varying_param_name} values", position=0, leave=True, ncols=80):
        all_results[vary_val] = {
            'energies_idmrg': [],
            'fillings_idmrg': [],
            'fixed_value': fixed_value,
            'states_retained': states_retained,
            'Nc': Nc
        }
        
        # Add finite DMRG storage if requested
        if include_finite_dmrg:
            all_results[vary_val]['energies_finite_dmrg'] = []
            all_results[vary_val]['fillings_finite_dmrg'] = []
        
        # Initialize storage for each int_sep configuration
        for int_sep in int_sep_list:
            int_sep_key = f'int_sep_{int_sep[0]}_{int_sep[1]}'
            all_results[vary_val][f'energies_{int_sep_key}'] = []
            all_results[vary_val][f'fillings_{int_sep_key}'] = []
        
        for x_val in tqdm(x_values, desc=f"  {x_param_name} ({varying_param_name}={vary_val:.2f})", position=1, leave=False, ncols=80):
            # Build parameter dict for current iteration
            params = {
                x_param_name: x_val,
                varying_param_name: vary_val,
                fixed_param_name: fixed_value
            }
            
            # Extract U, V, t from params (with defaults if not present)
            U = params.get('U', 0.0)
            V = params.get('V', 0.0)
            t = params.get('t', 1.0)
            
            mu_0 = U / 2  # Half-filling
            
            # Infinite DMRG calculation
        if include_idmrg:
            try:
                # iDMRG calculation (same for all int_sep, only depends on v_sep)
                meta = {
                    "method": "iDMRG",
                    "U": U,
                    "V": V,
                    "t": t,
                    "L": L,
                    "Nc": Nc,
                    "int_sep": None,
                    "v_sep": v_sep_ratio,
                    "super_cluster_size": None,
                }
                energy_idmrg, filling_idmrg, _ = time_call(
                    timing_recorder, meta, run_dmrg_method, U, mu_0, V, v_sep_ratio, t, L, chi
                )
                energy_idmrg_subtracted = energy_idmrg + mu_0 * filling_idmrg
                    
                all_results[vary_val]['energies_idmrg'].append(energy_idmrg_subtracted)
                all_results[vary_val]['fillings_idmrg'].append(filling_idmrg)
            except Exception as e:
                # If iDMRG fails, append NaN and record the failure
                all_results[vary_val]['energies_idmrg'].append(np.nan)
                all_results[vary_val]['fillings_idmrg'].append(np.nan)
                failed_calculations.append({
                    'method': 'iDMRG',
                    'params': {x_param_name: x_val, varying_param_name: vary_val, fixed_param_name: fixed_value},
                    'error': str(e)
                })
            else:
                # If not including iDMRG, just append NaN
                all_results[vary_val]['energies_idmrg'].append(np.nan)
                all_results[vary_val]['fillings_idmrg'].append(np.nan)
            
            # Finite DMRG calculation (optional)
            if include_finite_dmrg:
                try:
                    # Finite DMRG calculation (with actual system size L)
                    meta = {
                        "method": "DMRG",
                        "U": U,
                        "V": V,
                        "t": t,
                        "L": L,
                        "Nc": Nc,
                        "int_sep": None,
                        "v_sep": v_sep_ratio,
                        "super_cluster_size": None,
                    }
                    energy_finite, _, filling_finite = time_call(
                        timing_recorder, meta, get_gnd, L, chi, U, t, mu_0, V, v_sep_ratio
                    )
                    energy_finite_per_site = energy_finite / L
                    # filling_finite is already per-site from get_gnd
                    energy_finite_subtracted = energy_finite_per_site + mu_0 * filling_finite
                    
                    all_results[vary_val]['energies_finite_dmrg'].append(energy_finite_subtracted)
                    all_results[vary_val]['fillings_finite_dmrg'].append(filling_finite)
                except Exception as e:
                    # If finite DMRG fails, append NaN and record the failure
                    all_results[vary_val]['energies_finite_dmrg'].append(np.nan)
                    all_results[vary_val]['fillings_finite_dmrg'].append(np.nan)
                    failed_calculations.append({
                        'method': 'Finite DMRG',
                        'params': {x_param_name: x_val, varying_param_name: vary_val, fixed_param_name: fixed_value},
                        'error': str(e)
                    })
            
            # Calculate for each int_sep configuration
            for int_sep in int_sep_list:
                int_sep_key = f'int_sep_{int_sep[0]}_{int_sep[1]}'
                
                try:
                    physical_params = PhysicalParams(U=U, mu_0=mu_0, V=V, t=t)
                    run_config = ClusterModelConfig(
                        L=L,
                        int_cluster_size=Nc,
                        cluster_separation_ratio=int_sep,
                        V_separation_ratio=v_sep_ratio,
                        ham_lib='quspin',
                        physical_params=physical_params,
                        model_bc='periodic',
                        int_cluster_bc='periodic',
                        super_cluster_bc='periodic',
                        solver_method=solver_method,
                        states_retained=states_retained
                    )
                    super_cluster_size = None
                    if timing_recorder is not None:
                        try:
                            clusters_tmp = generate_clusters(L, Nc, int_sep, v_sep_ratio)
                            super_cluster_size = supercluster_size_from_clusters(clusters_tmp)
                        except Exception:
                            super_cluster_size = None
                    meta = {
                        "method": "cluster_ED",
                        "U": U,
                        "V": V,
                        "t": t,
                        "L": L,
                        "Nc": Nc,
                        "int_sep": int_sep,
                        "v_sep": v_sep_ratio,
                        "super_cluster_size": super_cluster_size,
                    }
                    system_expectations, _ = time_call(timing_recorder, meta, get_general_expectations, run_config, timing_recorder=timing_recorder)
                    energy, filling, _ = system_expectations
                    print(f"Raw Energy: {energy}, raw Filling: {filling}")
                    energy_subtracted = (energy + mu_0 * filling) / L
                    filling_per_site = filling / L
                    
                    all_results[vary_val][f'energies_{int_sep_key}'].append(energy_subtracted)
                    all_results[vary_val][f'fillings_{int_sep_key}'].append(filling_per_site)
                except Exception as e:
                    # If calculation fails, append NaN and record the failure
                    all_results[vary_val][f'energies_{int_sep_key}'].append(np.nan)
                    all_results[vary_val][f'fillings_{int_sep_key}'].append(np.nan)
                    failed_calculations.append({
                        'method': f'int_sep={int_sep}',
                        'params': {x_param_name: x_val, varying_param_name: vary_val, fixed_param_name: fixed_value},
                        'error': str(e)
                    })
    
    print("\n")  # Add spacing after progress bars
    
    # Create plots
    figures = create_int_sep_comparison_plots(
        x_values, varying_values, all_results, int_sep_list, v_sep_ratio, 
        output_dir, show_plots, param_names, fixed_value, states_retained, Nc, L
    )
    
    # Save results to pickle if requested
    if save_pickle:
        pickle_dir = 'large_files/runs/comparisons_general'
        os.makedirs(pickle_dir, exist_ok=True)
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create descriptive filename
        v_sep_str = f"{v_sep_ratio[0]}_{v_sep_ratio[1]}"
        pickle_filename = f"comparison_v_sep_{v_sep_str}_L{L}_Nc{Nc}_{timestamp}.pkl"
        pickle_path = os.path.join(pickle_dir, pickle_filename)
        
        # Prepare data to save
        pickle_data = {
            'all_results': all_results,
            'parameters': {
                'v_sep_ratio': v_sep_ratio,
                'int_sep_list': int_sep_list,
                x_param_name: x_values.tolist() if hasattr(x_values, 'tolist') else x_values,
                varying_param_name: varying_values.tolist() if hasattr(varying_values, 'tolist') else varying_values,
                fixed_param_name: fixed_value,
                'L': L,
                'Nc': Nc,
                'chi': chi,
                'solver_method': solver_method,
                'states_retained': states_retained,
                'param_names': param_names
            },
            'timestamp': timestamp
        }
        
        # Save to pickle file
        with open(pickle_path, 'wb') as f:
            pickle.dump(pickle_data, f)
        
        print(f"\nSaved results to: {pickle_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    if include_idmrg or include_finite_dmrg:
        print("Summary Statistics (Mean Absolute Differences)")
    else:
        print("Summary Statistics (Cluster Method Comparisons)")
    print("=" * 60)
    
    if include_idmrg or include_finite_dmrg:
        for vary_val in varying_values:
            print(f"\n{varying_param_name}={vary_val:.2f}:")
            
            # Compare with iDMRG if available
            if include_idmrg:
                energies_idmrg = np.array(all_results[vary_val]['energies_idmrg'])
                if not np.all(np.isnan(energies_idmrg)):
                    for int_sep in int_sep_list:
                        int_sep_key = f'int_sep_{int_sep[0]}_{int_sep[1]}'
                        energies_method = np.array(all_results[vary_val][f'energies_{int_sep_key}'])
                        mae = np.nanmean(np.abs(energies_method - energies_idmrg))
                        print(f"  int_sep={int_sep} vs iDMRG: MAE={mae:.6f}")
            
            # Compare with finite DMRG if available
            if include_finite_dmrg:
                energies_finite = np.array(all_results[vary_val]['energies_finite_dmrg'])
                if not np.all(np.isnan(energies_finite)):
                    for int_sep in int_sep_list:
                        int_sep_key = f'int_sep_{int_sep[0]}_{int_sep[1]}'
                        energies_method = np.array(all_results[vary_val][f'energies_{int_sep_key}'])
                        mae = np.nanmean(np.abs(energies_method - energies_finite))
                        print(f"  int_sep={int_sep} vs Finite DMRG: MAE={mae:.6f}")
    else:
        # When no DMRG, compare cluster methods to each other
        for vary_val in varying_values:
            print(f"\n{varying_param_name}={vary_val:.2f}:")
            # Get energies for all methods
            method_energies = {}
            for int_sep in int_sep_list:
                int_sep_key = f'int_sep_{int_sep[0]}_{int_sep[1]}'
                method_energies[int_sep] = np.array(all_results[vary_val][f'energies_{int_sep_key}'])
            
            # Compare first method to others
            if len(int_sep_list) > 1:
                ref_method = int_sep_list[0]
                ref_energies = method_energies[ref_method]
                for int_sep in int_sep_list[1:]:
                    mae = np.nanmean(np.abs(method_energies[int_sep] - ref_energies))
                    print(f"  {int_sep} vs {ref_method}: MAE={mae:.6f}")
    
    # Report failed calculations
    if failed_calculations:
        print("\n" + "=" * 60)
        print("WARNING: Some calculations failed to converge")
        print("=" * 60)
        
        # Group failures by method
        failures_by_method = {}
        for failure in failed_calculations:
            method = failure['method']
            if method not in failures_by_method:
                failures_by_method[method] = []
            failures_by_method[method].append(failure)
        
        for method, failures in failures_by_method.items():
            print(f"\n{method}:")
            for failure in failures:
                params_str = ', '.join([f"{k}={v}" for k, v in failure['params'].items()])
                error_msg = failure['error'].split('\n')[0]  # Just first line of error
                print(f"  Failed at {params_str}")
                print(f"    Error: {error_msg}")
        
        print(f"\nTotal failed calculations: {len(failed_calculations)}")
        print("Note: Failed points are marked as NaN in the results and excluded from plots")
    
    if include_timing and timing_recorder is not None:
        all_results['_timings'] = timing_recorder.records
        if include_timing_plot and timing_recorder.records:
            fig_timing, timing_artifacts = plot_timings(
                timing_recorder.records,
                output_dir=output_dir,
                filename_prefix="timing_int_seps",
                show_plots=show_plots,
            )
            all_results['_timing_plot'] = timing_artifacts

    return figures, all_results


def create_int_sep_comparison_plots(
    x_values, varying_values, all_results, int_sep_list, v_sep_ratio,
    output_dir='large_files/plots', show_plots=True, param_names=None,
    fixed_value=None, states_retained=None, Nc=None, L=None
):
    """Create line plots comparing different int_sep configurations with DMRG."""
    
    # Handle backward compatibility
    if param_names is None:
        param_names = {'x': 'U', 'varying': 'V', 'fixed': 't'}
    if fixed_value is None:
        fixed_value = all_results[varying_values[0]].get('fixed_value', 'N/A')
    if states_retained is None:
        states_retained = all_results[varying_values[0]].get('states_retained', 'N/A')
    if Nc is None:
        Nc = all_results[varying_values[0]].get('Nc', 'N/A')
    if L is None:
        L = 20  # Default value
    
    # Group varying values into chunks of 3
    n_v_per_fig = 3
    n_figures = np.ceil(len(varying_values) / n_v_per_fig).astype(int)
    figures = []
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define colors for different methods (reserve red for DMRG)
    color_palette = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
    colors = {'DMRG': 'red'}
    
    for idx, int_sep in enumerate(int_sep_list):
        int_sep_key = f'int_sep {format_sep_as_pi(int_sep)}'
        colors[int_sep_key] = color_palette[idx % len(color_palette)]
    
    for fig_idx in range(n_figures):
        start_idx = fig_idx * n_v_per_fig
        end_idx = min(start_idx + n_v_per_fig, len(varying_values))
        current_varying_values = varying_values[start_idx:end_idx]
        n_cols = len(current_varying_values)
        
        # Create subplot titles
        energy_titles = [f'Energy/site vs {param_names["x"]} ({param_names["varying"]}={v:.2f})' for v in current_varying_values]
        filling_titles = [f'Filling/site vs {param_names["x"]} ({param_names["varying"]}={v:.2f})' for v in current_varying_values]
        
        fig = make_subplots(
            rows=2, cols=n_cols,
            subplot_titles=energy_titles + filling_titles,
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        for col_idx, vary_val in enumerate(current_varying_values):
            col = col_idx + 1
            
            # Check if iDMRG data exists and is not all NaN
            idmrg_energies = all_results[vary_val]['energies_idmrg']
            has_idmrg_data = not np.all(np.isnan(idmrg_energies))
            
            if has_idmrg_data:
                # Plot iDMRG results
                # Energy plot (top row)
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=idmrg_energies,
                        mode='lines+markers',
                        name='iDMRG',
                        legendgroup='idmrg',
                        legendgrouptitle=dict(text='DMRG') if col_idx == 0 else None,
                        line=dict(color=colors['DMRG'], width=3),
                        marker=dict(size=8, symbol='diamond'),
                        showlegend=(col_idx == 0)
                    ),
                    row=1, col=col
                )
                
                # Filling plot (bottom row)
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=all_results[vary_val]['fillings_idmrg'],
                        mode='lines+markers',
                        name='iDMRG',
                        legendgroup='idmrg',
                        line=dict(color=colors['DMRG'], width=3),
                        marker=dict(size=8, symbol='diamond'),
                        showlegend=False
                    ),
                    row=2, col=col
                )
            
            # Check if finite DMRG data exists
            if 'energies_finite_dmrg' in all_results[vary_val]:
                finite_dmrg_energies = all_results[vary_val]['energies_finite_dmrg']
                has_finite_dmrg_data = not np.all(np.isnan(finite_dmrg_energies))
                
                if has_finite_dmrg_data:
                    # Plot finite DMRG results
                    # Energy plot (top row)
                    fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=finite_dmrg_energies,
                        mode='lines+markers',
                        name=f'Finite DMRG (L={L})',
                        legendgroup='finite_dmrg',
                        legendgrouptitle=dict(text='Finite DMRG') if col_idx == 0 else None,
                        line=dict(color='magenta', width=2, dash='dash'),
                        marker=dict(size=6, symbol='triangle-up'),
                        showlegend=(col_idx == 0)
                    ),
                    row=1, col=col
                    )
                    
                    # Filling plot (bottom row)
                    fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=all_results[vary_val]['fillings_finite_dmrg'],
                        mode='lines+markers',
                        name=f'Finite DMRG (L={L})',
                        legendgroup='finite_dmrg',
                        line=dict(color='magenta', width=2, dash='dash'),
                        marker=dict(size=6, symbol='triangle-up'),
                        showlegend=False
                    ),
                    row=2, col=col
                    )
            
            # Plot each int_sep configuration
            for idx, int_sep in enumerate(int_sep_list):
                int_sep_key = f'int_sep_{int_sep[0]}_{int_sep[1]}'
                label = f'int_sep {format_sep_as_pi(int_sep)}'
                
                # Check if int_sep matches v_sep for marker selection
                if int_sep == v_sep_ratio:
                    marker_symbol = 'diamond'  # Same as DMRG
                else:
                    marker_symbol = ['circle', 'square', 'triangle-up', 'x', 'cross'][idx % 5]
                
                # Energy plot (top row)
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=all_results[vary_val][f'energies_{int_sep_key}'],
                        mode='lines+markers',
                        name=label,
                        legendgroup=int_sep_key,
                        legendgrouptitle=dict(text='Cluster ED') if (col_idx == 0 and idx == 0) else None,
                        line=dict(
                            color=colors[label],
                            width=2
                            # Removed dash parameter - all lines are solid now
                        ),
                        marker=dict(size=6, symbol=marker_symbol),
                        showlegend=(col_idx == 0)
                    ),
                    row=1, col=col
                )
                
                # Filling plot (bottom row)
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=all_results[vary_val][f'fillings_{int_sep_key}'],
                        mode='lines+markers',
                        name=label,
                        legendgroup=int_sep_key,
                        line=dict(
                            color=colors[label],
                            width=2
                            # Removed dash parameter - all lines are solid now
                        ),
                        marker=dict(size=6, symbol=marker_symbol),
                        showlegend=False
                    ),
                    row=2, col=col
                )
            
            # Update axes labels
            fig.update_xaxes(title_text=param_names['x'], row=1, col=col)
            fig.update_xaxes(title_text=param_names['x'], row=2, col=col)
            
            # Set y-axis range for filling plots
            fig.update_yaxes(range=[0, 2], row=2, col=col)
            
            if col == 1:
                fig.update_yaxes(title_text='Energy/site', row=1, col=col)
                fig.update_yaxes(title_text='Filling/site', row=2, col=col)
        
        # Update layout
        fixed_param_info = f"{param_names['fixed']}={fixed_value}"
        
        # Check which DMRG types are present
        has_idmrg = any(not np.all(np.isnan(all_results[v]['energies_idmrg'])) 
                       for v in current_varying_values if v in all_results)
        has_finite_dmrg = any('energies_finite_dmrg' in all_results[v] and 
                             not np.all(np.isnan(all_results[v]['energies_finite_dmrg']))
                             for v in current_varying_values if v in all_results)
        
        # Build title based on what's included
        if has_idmrg and has_finite_dmrg:
            dmrg_label = 'iDMRG & Finite DMRG'
        elif has_idmrg:
            dmrg_label = 'iDMRG'
        elif has_finite_dmrg:
            dmrg_label = 'Finite DMRG'
        else:
            dmrg_label = None
        
        if dmrg_label:
            title_text = (f'Cluster Separation Comparison with {dmrg_label}<br>'
                         f'<sub>v_sep={format_sep_as_pi(v_sep_ratio)}, {fixed_param_info}, '
                         f'Nc={Nc}, states={states_retained}, L={L} | Page {fig_idx+1}/{n_figures}</sub>')
        else:
            title_text = (f'Cluster Separation Comparison<br>'
                         f'<sub>v_sep={format_sep_as_pi(v_sep_ratio)}, {fixed_param_info}, '
                         f'Nc={Nc}, states={states_retained}, L={L} | Page {fig_idx+1}/{n_figures}</sub>')
        
        fig.update_layout(
            title=dict(text=title_text, x=0.5, xanchor='center'),
            height=700,
            width=400 * n_cols + 150,  # Extra width for right-side legend
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02
            ),
            hovermode='x unified'
        )
        
        # Save and/or show
        v_sep_str = f"{v_sep_ratio[0]}_{v_sep_ratio[1]}"
        fixed_str = f"{param_names['fixed']}_{fixed_value}".replace('.', 'p')
        filename = f'int_sep_comparison_v_sep_{v_sep_str}_{fixed_str}_Nc{Nc}_page_{fig_idx+1}.html'
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath)
        print(f"Saved figure to {filepath}")
        
        if show_plots:
            fig.show()
        
        figures.append(fig)
    
    return figures


def create_u_value_comparison_plots(
    u_list: Sequence[float],
    v_list: Sequence[float],
    cluster_sizes: Sequence[int],
    cluster_results: Dict[Tuple[int, float], np.ndarray],
    cluster_fillings: Dict[Tuple[int, float], np.ndarray],
    idmrg_cache: np.ndarray,
    finite_dmrg_cache: np.ndarray,
    idmrg_fill_cache: np.ndarray,
    finite_dmrg_fill_cache: np.ndarray,
    *,
    v_sep_ratio: Tuple[int, int],
    t: float,
    L: int,
    chi: int,
    states_retained: int,
    color_map: Dict[int, str],
    output_dir: str,
    filename_prefix: str,
    timestamp: str,
    show_plots: bool,
    save_html: bool,
    set_filling: Optional[float] = None,
    mu_source_Nc: Optional[int] = None,
    dmrg_fixed_filling: bool = False,
    plot_relative_error: bool = False,
):
    """Create paginated U-value comparison plots with up to 3 V panels per page."""
    os.makedirs(output_dir, exist_ok=True)

    n_v_per_fig = 3
    n_figures = int(np.ceil(len(v_list) / n_v_per_fig))
    figures: List[go.Figure] = []
    html_paths: List[str] = []
    v_to_index = {V: idx for idx, V in enumerate(v_list)}

    for fig_idx in range(n_figures):
        start = fig_idx * n_v_per_fig
        end = min(start + n_v_per_fig, len(v_list))
        current_vs = v_list[start:end]
        n_cols = len(current_vs)

        energy_titles = [("Relative error" if plot_relative_error else "Energy") + f": V = {V:.3g}" for V in current_vs]
        filling_titles = [f"Filling: V = {V:.3g}" for V in current_vs]

        fig = make_subplots(
            rows=2,
            cols=n_cols,
            subplot_titles=energy_titles + filling_titles,
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )

        any_trace = False
        for col_idx, V in enumerate(current_vs):
            col = col_idx + 1
            v_idx = v_to_index[V]
            row_energy, row_filling = 1, 2

            reference_series = finite_dmrg_cache[:, v_idx] if plot_relative_error else None
            if plot_relative_error:
                if not np.any(np.isfinite(reference_series)):
                    raise ValueError(f"No finite DMRG reference available to compute relative errors for V={V}.")

                def rel_err(series: np.ndarray) -> np.ndarray:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        return np.where(
                            np.isfinite(series) & np.isfinite(reference_series) & (reference_series != 0),
                            np.abs(series - reference_series) / np.abs(reference_series),
                            np.nan,
                        )

            idmrg_energies = idmrg_cache[:, v_idx]
            if np.any(np.isfinite(idmrg_energies)) and not plot_relative_error:
                if plot_relative_error:
                    idmrg_rel = rel_err(idmrg_energies)
                    fig.add_trace(
                        go.Scatter(
                            x=u_list,
                            y=idmrg_rel,
                            mode='lines+markers',
                            name="iDMRG error",
                            legendgroup="iDMRG",
                            marker=dict(color='black', size=6, symbol='x'),
                            line=dict(color='black', width=2, dash='dash'),
                            showlegend=(col_idx == 0),
                            hovertemplate="U=%{x:.3g}<br>rel_err=%{y:.2%}<extra></extra>",
                        ),
                        row=row_energy,
                        col=col,
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=u_list,
                            y=idmrg_energies,
                            mode='lines+markers',
                            name="iDMRG",
                            legendgroup="iDMRG",
                            marker=dict(color='black', size=6, symbol='x'),
                            line=dict(color='black', width=2, dash='dash'),
                            showlegend=(col_idx == 0),
                            hovertemplate="U=%{x:.3g}<br>E_iDMRG=%{y:.6f}<extra></extra>",
                        ),
                        row=row_energy,
                        col=col,
                    )
                any_trace = True

            idmrg_fills = idmrg_fill_cache[:, v_idx]
            if np.any(np.isfinite(idmrg_fills)):
                fig.add_trace(
                    go.Scatter(
                        x=u_list,
                        y=idmrg_fills,
                        mode='lines+markers',
                        name="iDMRG (fill)",
                        legendgroup="iDMRG_fill",
                        marker=dict(color='black', size=6, symbol='x'),
                        line=dict(color='black', width=2, dash='dash'),
                        showlegend=False,
                        hovertemplate="U=%{x:.3g}<br>n_iDMRG=%{y:.6f}<extra></extra>",
                    ),
                    row=row_filling,
                    col=col,
                )
                any_trace = True

            finite_dmrg_energies = finite_dmrg_cache[:, v_idx]
            if np.any(np.isfinite(finite_dmrg_energies)) and not plot_relative_error:
                finite_label = "Finite DMRG (fixed N)" if dmrg_fixed_filling else "Finite DMRG"
                fig.add_trace(
                    go.Scatter(
                        x=u_list,
                        y=finite_dmrg_energies,
                        mode='lines+markers',
                        name=finite_label,
                        legendgroup=finite_label,
                        marker=dict(color='gray', size=6, symbol='cross'),
                        line=dict(color='gray', width=2, dash='dot'),
                        showlegend=(col_idx == 0),
                        hovertemplate="U=%{x:.3g}<br>E_Finite=%{y:.6f}<extra></extra>",
                    ),
                    row=row_energy,
                    col=col,
                )
                any_trace = True

            finite_dmrg_fills = finite_dmrg_fill_cache[:, v_idx]
            if np.any(np.isfinite(finite_dmrg_fills)):
                finite_fill_label = "Finite DMRG (fill, fixed N)" if dmrg_fixed_filling else "Finite DMRG (fill)"
                fig.add_trace(
                    go.Scatter(
                        x=u_list,
                        y=finite_dmrg_fills,
                        mode='lines+markers',
                        name=finite_fill_label,
                        legendgroup=finite_fill_label,
                        marker=dict(color='gray', size=6, symbol='cross'),
                        line=dict(color='gray', width=2, dash='dot'),
                        showlegend=False,
                        hovertemplate="U=%{x:.3g}<br>n_Finite=%{y:.6f}<extra></extra>",
                    ),
                    row=row_filling,
                    col=col,
                )
                any_trace = True

            for Nc in cluster_sizes:
                cluster_energies = cluster_results[(Nc, V)]
                cluster_fills = cluster_fillings[(Nc, V)]
                xs, ys, hover_text = [], [], []
                xs_fill, ys_fill = [], []

                for u_idx, U in enumerate(u_list):
                    c_en = cluster_energies[u_idx]
                    if not np.isfinite(c_en):
                        continue

                    # Always collect filling data
                    c_fill = cluster_fills[u_idx]
                    if np.isfinite(c_fill):
                        xs_fill.append(U)
                        ys_fill.append(c_fill)

                    if plot_relative_error:
                        ref = reference_series[u_idx]
                        if not (np.isfinite(ref) and ref != 0):
                            continue
                        c_err = np.abs(c_en - ref) / np.abs(ref)
                        xs.append(U)
                        ys.append(c_err)
                        hover_text.append(
                            f"U={U:.3g}<br>V={V:.3g}<br>Nc={Nc}<br>rel_err={c_err:.2%}"
                        )
                    else:
                        xs.append(U)
                        ys.append(c_en)
                        hover_text.append(
                            f"U={U:.3g}<br>V={V:.3g}<br>Nc={Nc}<br>E_cluster={c_en:.6f}"
                        )

                if not xs:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode='lines+markers',
                        name=f"Nc={Nc}",
                        legendgroup=f"Nc={Nc}",
                        marker=dict(color=color_map[Nc], size=8),
                        line=dict(color=color_map[Nc], width=2),
                        showlegend=(col_idx == 0),
                        hovertemplate="%{text}<extra></extra>",
                        text=hover_text,
                    ),
                    row=row_energy,
                    col=col,
                )
                any_trace = True

                if xs_fill:
                    fig.add_trace(
                        go.Scatter(
                            x=xs_fill,
                            y=ys_fill,
                            mode='lines+markers',
                            name=f"Nc={Nc} (fill)",
                            legendgroup=f"Nc_fill_{Nc}",
                            marker=dict(color=color_map[Nc], size=8, symbol='circle-open'),
                            line=dict(color=color_map[Nc], width=2, dash='dot'),
                            showlegend=False,
                            hovertemplate="U=%{x:.3g}<br>n_cluster=%{y:.6f}<extra></extra>",
                        ),
                        row=row_filling,
                        col=col,
                    )
                    any_trace = True

            # Add vertical line at U=V/2 for both energy and filling subplots
            u_critical = V / 2

            # Add as a trace for legend (only on first column to show in legend once)
            for row in [row_energy, row_filling]:
                # Add invisible scatter trace for legend entry (only once per figure)
                if col == 1 and row == row_energy:
                    fig.add_trace(
                        go.Scatter(
                            x=[u_critical],
                            y=[None],
                            mode='lines',
                            name='U=V/2',
                            line=dict(color='gray', dash='dot', width=1),
                            showlegend=True,
                            hoverinfo='skip',
                        ),
                        row=row,
                        col=col,
                    )

                # Add the actual vertical line
                fig.add_vline(
                    x=u_critical,
                    row=row,
                    col=col,
                    line_dash="dot",
                    line_color="gray",
                    line_width=1,
                    annotation_text=None,  # No annotation to avoid clutter
                )

            fig.update_xaxes(title_text="U", row=row_energy, col=col)
            fig.update_xaxes(title_text="U", row=row_filling, col=col)
            fig.update_yaxes(
                title_text="Relative error" if plot_relative_error else "Energy per site",
                row=row_energy,
                col=col,
                type='linear',
                tickformat='.2%' if plot_relative_error else '.4f',
            )
            fig.update_yaxes(title_text="Filling per site", row=row_filling, col=col, type='linear', range=[0, 2])

        if not any_trace:
            raise RuntimeError("No valid data points available to plot U-value convergence.")

        v_sep_label = format_sep_as_pi(v_sep_ratio)
        annotation_parts = [
            f"v_sep={v_sep_label}",
            f"t={t}",
            f"L={L}",
            f"chi={chi}",
            f"states={states_retained}",
        ]
        if set_filling is not None:
            annotation_parts.insert(1, f"n_target={set_filling}")
            if mu_source_Nc is not None:
                annotation_parts.insert(2, f"mu_from=Nc{mu_source_Nc}")
            if dmrg_fixed_filling:
                annotation_parts.insert(3, "finite_DMRG=fixed_N")
        annotation_text = ", ".join(annotation_parts) + f" | Page {fig_idx + 1}/{n_figures}"

        fig.update_layout(
            title=dict(
                text="Relative Error vs U" if plot_relative_error else "Ground State Energy vs U",
                x=0.5,
                xanchor='center'
            ),
            hovermode='closest',
            legend_title="Method",
        )
        fig.add_annotation(
            text=annotation_text,
            x=0.5,
            xref='paper',
            y=1.06,
            yref='paper',
            showarrow=False,
            font=dict(size=12, color='gray'),
        )

        if save_html:
            html_name = f"{filename_prefix}_L{L}_chi{chi}_{timestamp}_page_{fig_idx + 1}.html"
            html_path = os.path.join(output_dir, html_name)
            fig.write_html(html_path)
            html_paths.append(html_path)
            print(f"Saved figure to {html_path}")

        if show_plots:
            fig.show()

        figures.append(fig)

    return figures, html_paths


def compare_cluster_sizes_with_dmrg(
    v_sep_ratio: Tuple[int, int],
    int_sep_ratios: Union[Tuple[int, int], Dict[int, Tuple[int, int]]],
    cluster_sizes: Sequence[int],
    U_values: Sequence[float],
    V_values: Sequence[float],
    *,
    t: float = 1.0,
    L: int = 20,
    chi: int = 32,
    solver_method: str = 'dense_ED',
    states_retained: int = 4,
    output_dir: str = 'large_files/plots',
    show_plots: bool = True,
    save_html: bool = True,
    save_pickle: bool = True,
    filename_prefix: str = 'cluster_size_convergence',
    log_yaxis: bool = True,
    reference_scheme: str = 'idmrg',
    results: Optional[Union[Dict, str, os.PathLike]] = None,
    include_timing: bool = False,
    include_timing_plot: bool = False,
) -> Tuple[go.Figure, Dict]:
    """
    Plot how the cluster method converges to iDMRG as the cluster size (N_c) increases.

    Each subplot fixes a value of U. Within that subplot the x-axis enumerates
    the provided cluster sizes and each line corresponds to a different V value.
    The y-axis shows |E_DMRG - E_cluster| / |E_DMRG|.

    Args:
        v_sep_ratio: Ratio controlling the AA modulation for V (passed to both
            cluster calculations and DMRG).
        int_sep_ratios: Either a single (p, q) tuple applied to every cluster
            size or a dict mapping each N_c to its specific interaction
            separation ratio.
        cluster_sizes: Iterable of cluster sizes (N_c) to test.
        U_values: Iterable of U values; each becomes its own subplot.
        V_values: Iterable of V strengths; each becomes a separate line.
        t: Hopping parameter.
        L: System size used for the cluster method.
        chi: Bond dimension for iDMRG.
        solver_method: Diagonalisation backend for the cluster Hamiltonian.
        states_retained: Number of states retained in the cluster solver.
        output_dir: Directory for saved artifacts.
        show_plots: Whether to open the generated Plotly figure.
        save_html: If True, save the interactive figure as HTML.
        save_pickle: If True, pickle the raw numerical results.
        filename_prefix: Prefix for saved artifact names.
        log_yaxis: Plot the y-axis on a log scale to highlight asymptotics.
        reference_scheme: Either 'idmrg' (default) or 'finite_dmrg' to choose
            which DMRG variant provides the reference energies.

    Returns:
        (figure, results_dict)
    """

    if t is None:
        raise ValueError("Parameter t must be specified for the cluster calculations.")

    def _coerce_ratio(value, label: str) -> Tuple[int, int]:
        if value is None:
            raise ValueError(f"{label} ratio must be provided.")
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{label} ratio must be a length-2 iterable, got {value!r}.")
        try:
            p = int(round(value[0]))
            q = int(round(value[1]))
        except Exception as exc:
            raise ValueError(f"Could not parse {label} ratio {value!r} into integers.") from exc
        if q == 0:
            raise ValueError(f"Denominator for {label} ratio cannot be zero.")
        return (p, q)

    cluster_sizes = sorted({int(size) for size in cluster_sizes})
    if not cluster_sizes:
        raise ValueError("Provide at least one cluster size (N_c).")

    U_values = np.asarray(U_values, dtype=float)
    V_values = np.asarray(V_values, dtype=float)
    if U_values.ndim != 1 or U_values.size == 0:
        raise ValueError("U_values must be a 1-D array with at least one entry.")
    if V_values.ndim != 1 or V_values.size == 0:
        raise ValueError("V_values must be a 1-D array with at least one entry.")

    u_list = [float(u) for u in U_values]
    v_list = [float(v) for v in V_values]

    if plot_relative_error and not include_finite_dmrg:
        raise ValueError("plot_relative_error requires include_finite_dmrg=True to supply the finite DMRG reference.")

    v_sep_ratio = _coerce_ratio(v_sep_ratio, "V separation")

    reference_scheme = reference_scheme.lower()
    if reference_scheme not in {'idmrg', 'finite_dmrg'}:
        raise ValueError("reference_scheme must be either 'idmrg' or 'finite_dmrg'.")

    if isinstance(int_sep_ratios, dict):
        ratio_map: Dict[int, Tuple[int, int]] = {}
        fallback_ratio: Optional[Tuple[int, int]] = None
        for v in int_sep_ratios.values():
            fallback_ratio = _coerce_ratio(v, "int_sep (fallback)")
            break
        for Nc in cluster_sizes:
            if Nc in int_sep_ratios:
                ratio_map[Nc] = _coerce_ratio(int_sep_ratios[Nc], f"int_sep (Nc={Nc})")
            else:
                if fallback_ratio is None:
                    raise ValueError(f"No int_sep ratio provided for cluster size Nc={Nc}.")
                warnings.warn(f"No int_sep ratio provided for Nc={Nc}; using fallback {fallback_ratio}.")
                ratio_map[Nc] = fallback_ratio
    else:
        common_ratio = _coerce_ratio(int_sep_ratios, "int_sep")
        ratio_map = {Nc: common_ratio for Nc in cluster_sizes}

    cluster_results: Dict[Tuple[int, float], np.ndarray]
    dmrg_cache: np.ndarray
    failed_calculations: List[Dict] = []
    timing_recorder = TimingRecorder() if include_timing else None

    if results is not None:
        if isinstance(results, (str, os.PathLike)):
            results_path = Path(results)
            if not results_path.exists():
                raise ValueError(f"Results file not found: {results_path}")
            with open(results_path, 'rb') as fh:
                results = pickle.load(fh)
        elif not isinstance(results, dict):
            raise ValueError("results must be a dict or path-like object when provided.")

        save_pickle = False  # Avoid re-saving when plotting from cached data
        print("Using precomputed results payload; skipping new simulations.")
        params = results.get('parameters', {})
        dmrg_cache = np.asarray(results.get('dmrg_energies', []), dtype=float)
        serialized_clusters = results.get('cluster_energies', {})
        cluster_results = {}

        if dmrg_cache.shape != (len(u_list), len(v_list)):
            raise ValueError("Provided dmrg_energies shape does not match U and V grids.")

        for Nc in cluster_sizes:
            cluster_by_v = serialized_clusters.get(str(Nc)) or serialized_clusters.get(Nc)
            if cluster_by_v is None:
                raise ValueError(f"Results payload missing data for cluster size Nc={Nc}.")
            for V in v_list:
                series = cluster_by_v.get(str(V)) or cluster_by_v.get(V)
                if series is None:
                    raise ValueError(f"Results payload missing data for V={V} at Nc={Nc}.")
                arr = np.asarray(series, dtype=float)
                if arr.size != len(u_list):
                    raise ValueError(f"Cluster series for Nc={Nc}, V={V} has length {arr.size}, expected {len(u_list)}.")
                cluster_results[(Nc, V)] = arr

        # Override metadata from results where available
        stored_ratio_map = results.get('int_sep_ratios')
        if stored_ratio_map:
            converted_ratio_map = {}
            for key, val in stored_ratio_map.items():
                try:
                    Nc_key = int(key)
                except (TypeError, ValueError):
                    Nc_key = key
                converted_ratio_map[Nc_key] = _coerce_ratio(val, f"int_sep (Nc={Nc_key})")
            ratio_map = converted_ratio_map
        params_v_sep = results.get('parameters', {}).get('v_sep_ratio', v_sep_ratio)
        v_sep_ratio = _coerce_ratio(params_v_sep, "V separation")
        t = params.get('t', t)
        L = params.get('L', L)
        chi = params.get('chi', chi)
        states_retained = params.get('states_retained', states_retained)
        reference_scheme = params.get('reference_scheme', reference_scheme)
    else:
        # Storage for computed energies
        cluster_results = {
            (Nc, V): np.full(len(u_list), np.nan, dtype=float)
            for Nc in cluster_sizes
            for V in v_list
        }
        dmrg_cache = np.full((len(u_list), len(v_list)), np.nan, dtype=float)

        print("=" * 60)
        ref_label = "iDMRG" if reference_scheme == 'idmrg' else "finite DMRG"
        print(f"Computing {ref_label} reference energies")
        print("=" * 60)
        for u_idx, U in enumerate(u_list):
            mu_0 = U / 2.0
            for v_idx, V in enumerate(v_list):
                try:
                    if reference_scheme == 'idmrg':
                        meta = {
                            "method": "iDMRG",
                            "U": U,
                            "V": V,
                            "t": t,
                            "L": L,
                            "Nc": None,
                            "int_sep": None,
                            "v_sep": v_sep_ratio,
                            "super_cluster_size": None,
                        }
                        energy_dmrg, filling_dmrg, _ = time_call(
                            timing_recorder,
                            meta,
                            run_dmrg_method,
                            U,
                            mu_0,
                            V,
                            v_sep_ratio,
                            t,
                            L,
                            chi,
                        )
                        dmrg_value = energy_dmrg + mu_0 * filling_dmrg
                    else:
                        meta = {
                            "method": "DMRG",
                            "U": U,
                            "V": V,
                            "t": t,
                            "L": L,
                            "Nc": None,
                            "int_sep": None,
                            "v_sep": v_sep_ratio,
                            "super_cluster_size": None,
                        }
                        energy_finite, _, filling_finite = time_call(
                            timing_recorder,
                            meta,
                            get_gnd,
                            L,
                            chi,
                            U,
                            t,
                            mu_0,
                            V,
                            v_sep_ratio,
                        )
                        energy_finite_per_site = energy_finite / L
                        dmrg_value = energy_finite_per_site + mu_0 * filling_finite

                    dmrg_cache[u_idx, v_idx] = dmrg_value
                except Exception as exc:
                    import traceback
                    error_msg = str(exc) or f"{type(exc).__name__}: {repr(exc)}"
                    failed_calculations.append({
                        'method': ref_label,
                        'params': {'U': U, 'V': V},
                        'error': error_msg,
                        'traceback': traceback.format_exc(),
                    })

        print("\n")
        print("=" * 60)
        print("Running cluster calculations for each N_c")
        print("=" * 60)
        for Nc in tqdm(cluster_sizes, desc="Cluster sizes", ncols=80):
            int_sep_ratio = ratio_map[Nc]
            super_cluster_size = None
            if timing_recorder is not None:
                try:
                    clusters_tmp = generate_clusters(L, Nc, int_sep_ratio, v_sep_ratio)
                    super_cluster_size = supercluster_size_from_clusters(clusters_tmp)
                except Exception:
                    super_cluster_size = None
            for v_idx, V in enumerate(v_list):
                for u_idx, U in enumerate(u_list):
                    mu_0 = U / 2.0
                    physical_params = PhysicalParams(U=U, mu_0=mu_0, V=V, t=t)
                    run_config = ClusterModelConfig(
                        L=L,
                        int_cluster_size=Nc,
                        cluster_separation_ratio=int_sep_ratio,
                        V_separation_ratio=v_sep_ratio,
                        ham_lib='quspin',
                        physical_params=physical_params,
                        model_bc='periodic',
                        int_cluster_bc='periodic',
                        super_cluster_bc='periodic',
                        solver_method=solver_method,
                        states_retained=states_retained,
                    )
                    try:
                        meta = {
                            "method": "cluster_ED",
                            "U": U,
                            "V": V,
                            "t": t,
                            "L": L,
                            "Nc": Nc,
                            "int_sep": int_sep_ratio,
                            "v_sep": v_sep_ratio,
                            "super_cluster_size": super_cluster_size,
                        }
                        system_expectations, _ = time_call(timing_recorder, meta, get_general_expectations, run_config, timing_recorder=timing_recorder)
                        energy, filling, _ = system_expectations
                        energy_subtracted = (energy + mu_0 * filling) / L
                        cluster_results[(Nc, V)][u_idx] = energy_subtracted
                        cluster_fillings[(Nc, V)][u_idx] = filling / L
                    except Exception as exc:
                        import traceback
                        error_msg = str(exc) or f"{type(exc).__name__}: {repr(exc)}"
                        failed_calculations.append({
                            'method': f'cluster Nc={Nc}',
                            'params': {'U': U, 'V': V},
                            'error': error_msg,
                            'traceback': traceback.format_exc(),
                        })

    n_cols = min(3, len(u_list))
    n_rows = int(np.ceil(len(u_list) / n_cols))
    subplot_titles = [
        f"U = {u_list[idx]:.3g}" if idx < len(u_list) else ""
        for idx in range(n_rows * n_cols)
    ]
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf',
    ]
    color_map = {
        V: color_palette[idx % len(color_palette)]
        for idx, V in enumerate(v_list)
    }

    any_trace = False
    subplot_annotations: List[Dict] = []
    for u_idx, U in enumerate(u_list):
        row = (u_idx // n_cols) + 1
        col = (u_idx % n_cols) + 1
        for v_idx, V in enumerate(v_list):
            dmrg_energy = dmrg_cache[u_idx, v_idx]
            if not np.isfinite(dmrg_energy):
                continue

            xs = []
            ys = []
            hover_text = []
            for Nc in cluster_sizes:
                cluster_energy = cluster_results[(Nc, V)][u_idx]
                if not np.isfinite(cluster_energy):
                    continue
                denom = np.abs(dmrg_energy)
                if denom < 1e-12:
                    denom = 1e-12
                rel_err = np.abs(dmrg_energy - cluster_energy) / denom
                xs.append(Nc)
                ys.append(rel_err)
                hover_text.append(
                    f"U={U:.3g}<br>V={V:.3g}<br>Nc={Nc}<br>"
                    f"E_cluster={cluster_energy:.6f}<br>E_DMRG={dmrg_energy:.6f}"
                )

            if not xs:
                continue

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode='lines+markers',
                    name=f"V={V:.3g}",
                    legendgroup=f"V={V:.3g}",
                    marker=dict(color=color_map[V], size=8),
                    line=dict(color=color_map[V], width=2),
                    showlegend=(u_idx == 0),
                    hovertemplate="%{text}<br>|ΔE|/|E|=%{y:.3e}<extra></extra>",
                    text=hover_text,
                ),
                row=row,
                col=col,
            )
            any_trace = True

        fig.update_xaxes(title_text="Cluster size (N_c)", row=row, col=col)
        fig.update_yaxes(
            title_text="|E_DMRG - E_cluster| / |E_DMRG|",
            row=row,
            col=col,
            type='log' if log_yaxis else 'linear',
            tickformat='.2%'
        )

        x_center = (col - 0.5) / n_cols
        y_top = 1 - (row - 1) / n_rows
        subplot_annotations.append(
            dict(
                text=f"U = {U:.3g}",
                x=x_center,
                xref='paper',
                y=y_top - 0.06,
                yref='paper',
                showarrow=False,
                font=dict(size=12, color='black')
            )
        )

    if not any_trace:
        raise RuntimeError("No valid data points available to plot cluster-size convergence.")

    v_sep_label = format_sep_as_pi(v_sep_ratio)
    annotation_text = (
        f"v_sep={v_sep_label}, t={t}, L={L}, chi={chi}, states={states_retained}"
    )

    fig.update_layout(
        title=dict(text="Relative energy error vs cluster size", x=0.5, xanchor='center'),
        hovermode='closest',
        legend_title="V values",
        annotations=subplot_annotations + [
            dict(
                text=annotation_text,
                x=0.5,
                xref='paper',
                y=1.06,
                yref='paper',
                showarrow=False,
                font=dict(size=12, color='gray'),
            )
        ],
    )

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_paths = {}

    if save_html:
        html_name = f"{filename_prefix}_L{L}_chi{chi}_{timestamp}.html"
        html_path = os.path.join(output_dir, html_name)
        fig.write_html(html_path)
        saved_paths['html'] = html_path
        print(f"Saved figure to {html_path}")

    cluster_energy_serialized = {
        Nc: {V: cluster_results[(Nc, V)].tolist() for V in v_list}
        for Nc in cluster_sizes
    }
    results_payload = {
        'cluster_sizes': cluster_sizes,
        'U_values': u_list,
        'V_values': v_list,
        'cluster_energies': cluster_energy_serialized,
        'dmrg_energies': dmrg_cache.tolist(),
        'int_sep_ratios': {Nc: ratio_map[Nc] for Nc in cluster_sizes},
        'parameters': {
            'v_sep_ratio': v_sep_ratio,
            't': t,
            'L': L,
            'chi': chi,
            'solver_method': solver_method,
            'states_retained': states_retained,
            'reference_scheme': reference_scheme,
        },
        'artifacts': saved_paths,
        'failures': failed_calculations,
    }
    if include_timing and timing_recorder is not None:
        results_payload['timings'] = timing_recorder.records
        if include_timing_plot and timing_recorder.records:
            fig_timing, timing_artifacts = plot_timings(
                timing_recorder.records,
                output_dir=output_dir,
                filename_prefix=f"{filename_prefix}_timing",
                show_plots=show_plots,
            )
            saved_paths['timing_plot'] = timing_artifacts.get('html')

    if save_pickle:
        pickle_name = f"{filename_prefix}_L{L}_chi{chi}_{timestamp}.pkl"
        pickle_path = os.path.join(output_dir, pickle_name)
        with open(pickle_path, 'wb') as fh:
            pickle.dump(results_payload, fh)
        saved_paths['pickle'] = pickle_path
        print(f"Saved data to {pickle_path}")

    if failed_calculations:
        print("\n" + "=" * 60)
        print("WARNING: Some calculations failed")
        print("=" * 60)
        for failure in failed_calculations:
            params_desc = ', '.join(f"{k}={v}" for k, v in failure['params'].items())
            print(f"{failure['method']}: {params_desc}")
            error_lines = failure['error'].splitlines()
            print(f"  Error: {error_lines[0] if error_lines else '(no error message)'}")
            if 'traceback' in failure:
                print(f"  Full traceback:\n{failure['traceback']}")

    return fig, results_payload


if __name__ == "__main__":
    # Example usage
    L = 12
    Nc = 3
    U_values = np.linspace(0, 6, 7)
    V_values = np.array([1e-8, 0.5, 1.0, 2.0])
    v_sep_ratio = (1, 6)
    int_sep_list = [(1, 6), (1, 4), (1, 3)]
    
    figures, results = compare_int_seps_with_dmrg(
        v_sep_ratio=v_sep_ratio,
        int_sep_list=int_sep_list,
        U_values=U_values,
        V_values=V_values,
        L=L,
        Nc=Nc,
        solver_method='sparse_ED',
        chi=32,
        show_plots=True
    )


def create_u_value_rho_q_plots(
    u_list: Sequence[float],
    v_list: Sequence[float],
    cluster_sizes: Sequence[int],
    cluster_rho_q: Dict[Tuple[int, float], np.ndarray],
    finite_dmrg_rho_q: np.ndarray,
    *,
    v_sep_ratio: Tuple[int, int],
    t: float,
    L: int,
    chi: int,
    states_retained: int,
    color_map: Dict[int, str],
    output_dir: str,
    filename_prefix: str,
    timestamp: str,
    show_plots: bool,
    save_html: bool,
    set_filling: Optional[float] = None,
    mu_source_Nc: Optional[int] = None,
    dmrg_fixed_filling: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    n_v_per_fig = 3
    n_figures = int(np.ceil(len(v_list) / n_v_per_fig))
    figures: List[go.Figure] = []
    html_paths: List[str] = []
    v_to_index = {V: idx for idx, V in enumerate(v_list)}

    for fig_idx in range(n_figures):
        start = fig_idx * n_v_per_fig
        end = min(start + n_v_per_fig, len(v_list))
        current_vs = v_list[start:end]
        n_cols = len(current_vs)

        fig = make_subplots(
            rows=1,
            cols=n_cols,
            subplot_titles=[f"rho_Q: V = {V:.3g}" for V in current_vs],
            horizontal_spacing=0.08,
        )

        any_trace = False
        for col_idx, V in enumerate(current_vs):
            col = col_idx + 1
            v_idx = v_to_index[V]

            finite_series = finite_dmrg_rho_q[:, v_idx]
            if np.any(np.isfinite(finite_series)):
                finite_label = "Finite DMRG (fixed N)" if dmrg_fixed_filling else "Finite DMRG"
                fig.add_trace(
                    go.Scatter(
                        x=u_list,
                        y=finite_series,
                        mode='lines+markers',
                        name=finite_label,
                        legendgroup=finite_label,
                        marker=dict(color='gray', size=6, symbol='cross'),
                        line=dict(color='gray', width=2, dash='dot'),
                        showlegend=(col_idx == 0),
                        hovertemplate="U=%{x:.3g}<br>rho_Q=%{y:.6f}<extra></extra>",
                    ),
                    row=1,
                    col=col,
                )
                any_trace = True

            for Nc in cluster_sizes:
                cluster_series = cluster_rho_q[(Nc, V)]
                xs, ys, hover_text = [], [], []
                for u_idx, U in enumerate(u_list):
                    rho_q = cluster_series[u_idx]
                    if not np.isfinite(rho_q):
                        continue
                    xs.append(U)
                    ys.append(rho_q)
                    hover_text.append(
                        f"U={U:.3g}<br>V={V:.3g}<br>Nc={Nc}<br>rho_Q={rho_q:.6f}"
                    )

                if not xs:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode='lines+markers',
                        name=f"Nc={Nc}",
                        legendgroup=f"Nc={Nc}",
                        marker=dict(color=color_map[Nc], size=8),
                        line=dict(color=color_map[Nc], width=2),
                        showlegend=(col_idx == 0),
                        hovertemplate="%{text}<extra></extra>",
                        text=hover_text,
                    ),
                    row=1,
                    col=col,
                )
                any_trace = True

            u_critical = V / 2.0
            if col == 1:
                fig.add_trace(
                    go.Scatter(
                        x=[u_critical],
                        y=[None],
                        mode='lines',
                        name='U=V/2',
                        line=dict(color='gray', dash='dot', width=1),
                        showlegend=True,
                        hoverinfo='skip',
                    ),
                    row=1,
                    col=col,
                )

            fig.add_vline(
                x=u_critical,
                row=1,
                col=col,
                line_dash="dot",
                line_color="gray",
                line_width=1,
                annotation_text=None,
            )

            fig.update_xaxes(title_text="U", row=1, col=col)
            fig.update_yaxes(title_text="rho_Q", row=1, col=col, type='linear', tickformat='.4f')

        if not any_trace:
            raise RuntimeError("No valid data points available to plot rho_Q convergence.")

        v_sep_label = format_sep_as_pi(v_sep_ratio)
        annotation_parts = [
            f"v_sep={v_sep_label}",
            f"t={t}",
            f"L={L}",
            f"chi={chi}",
            f"states={states_retained}",
        ]
        if set_filling is not None:
            annotation_parts.insert(1, f"n_target={set_filling}")
            if mu_source_Nc is not None:
                annotation_parts.insert(2, f"mu_from=Nc{mu_source_Nc}")
            if dmrg_fixed_filling:
                annotation_parts.insert(3, "finite_DMRG=fixed_N")
        annotation_text = ", ".join(annotation_parts) + f" | Page {fig_idx + 1}/{n_figures}"

        fig.update_layout(
            title=dict(
                text="Density-Wave Amplitude vs U",
                x=0.5,
                xanchor='center',
            ),
            hovermode='closest',
            legend_title="Method",
        )
        fig.add_annotation(
            text=annotation_text,
            x=0.5,
            xref='paper',
            y=1.08,
            yref='paper',
            showarrow=False,
            font=dict(size=12, color='gray'),
        )

        if save_html:
            html_name = f"{filename_prefix}_L{L}_chi{chi}_{timestamp}_page_{fig_idx + 1}.html"
            html_path = os.path.join(output_dir, html_name)
            fig.write_html(html_path)
            html_paths.append(html_path)
            print(f"Saved figure to {html_path}")

        if show_plots:
            fig.show()

        figures.append(fig)

    return figures, html_paths


def compare_U_values_rhoQ_with_dmrg(
    v_sep_ratio: Tuple[int, int],
    int_sep_ratios: Union[Tuple[int, int], Dict[int, Tuple[int, int]]],
    cluster_sizes: Sequence[int],
    U_values: Sequence[float],
    V_values: Sequence[float],
    *,
    t: float = 1.0,
    L: int = 20,
    chi: int = 32,
    solver_method: str = 'dense_ED',
    states_retained: int = 4,
    output_dir: str = 'large_files/plots',
    show_plots: bool = True,
    save_html: bool = True,
    save_data: bool = True,
    filename_prefix: str = 'U_value_rho_q_comparison',
    set_filling: Optional[float] = None,
    dmrg_fixed_filling: bool = False,
    results: Optional[Union[Dict, str, os.PathLike]] = None,
) -> Tuple[go.Figure, Dict]:
    if t is None:
        raise ValueError("Parameter t must be specified for the cluster calculations.")
    if dmrg_fixed_filling and set_filling is None:
        raise ValueError("dmrg_fixed_filling=True requires set_filling to be provided.")

    def _coerce_ratio(value, label: str) -> Tuple[int, int]:
        if value is None:
            raise ValueError(f"{label} ratio must be provided.")
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{label} ratio must be a length-2 iterable, got {value!r}.")
        try:
            p = int(round(value[0]))
            q = int(round(value[1]))
        except Exception as exc:
            raise ValueError(f"Could not parse {label} ratio {value!r} into integers.") from exc
        if q == 0:
            raise ValueError(f"Denominator for {label} ratio cannot be zero.")
        return (p, q)

    cluster_sizes = sorted({int(size) for size in cluster_sizes})
    if not cluster_sizes:
        raise ValueError("Provide at least one cluster size (N_c).")

    U_values = np.asarray(U_values, dtype=float)
    V_values = np.asarray(V_values, dtype=float)
    if U_values.ndim != 1 or U_values.size == 0:
        raise ValueError("U_values must be a 1-D array with at least one entry.")
    if V_values.ndim != 1 or V_values.size == 0:
        raise ValueError("V_values must be a 1-D array with at least one entry.")

    u_list = [float(u) for u in U_values]
    v_list = [float(v) for v in V_values]
    v_sep_ratio = _coerce_ratio(v_sep_ratio, "V separation")

    if isinstance(int_sep_ratios, dict):
        ratio_map: Dict[int, Tuple[int, int]] = {}
        for Nc in cluster_sizes:
            if Nc not in int_sep_ratios:
                raise ValueError(f"No int_sep ratio provided for cluster size Nc={Nc}.")
            ratio_map[Nc] = _coerce_ratio(int_sep_ratios[Nc], f"int_sep (Nc={Nc})")
    else:
        common_ratio = _coerce_ratio(int_sep_ratios, "int_sep")
        ratio_map = {Nc: common_ratio for Nc in cluster_sizes}

    failed_calculations: List[Dict] = []
    mu_source_Nc = max(cluster_sizes) if set_filling is not None else None

    if results is not None:
        if isinstance(results, (str, os.PathLike)):
            results_path = Path(results)
            if not results_path.exists():
                raise ValueError(f"Results file not found: {results_path}")
            with open(results_path, 'rb') as fh:
                results = pickle.load(fh)
        elif not isinstance(results, dict):
            raise ValueError("results must be a dict or path-like object when provided.")

        save_data = False
        print("Using precomputed results payload; skipping new simulations.")
        params = results.get('parameters', {})
        stored_set_filling = params.get('set_filling', set_filling)
        set_filling = float(stored_set_filling) if stored_set_filling is not None else None
        dmrg_fixed_filling = bool(params.get('dmrg_fixed_filling', dmrg_fixed_filling))
        mu_source_Nc = max(cluster_sizes) if set_filling is not None else None

        finite_dmrg_rho_q = np.asarray(results.get('finite_dmrg_rho_q', []), dtype=float)
        if finite_dmrg_rho_q.size == 0:
            finite_dmrg_rho_q = np.full((len(u_list), len(v_list)), np.nan, dtype=float)

        serialized_clusters = results.get('cluster_rho_q', {})
        cluster_rho_q: Dict[Tuple[int, float], np.ndarray] = {}
        for Nc in cluster_sizes:
            cluster_by_v = serialized_clusters.get(str(Nc)) or serialized_clusters.get(Nc)
            if cluster_by_v is None:
                raise ValueError(f"Results payload missing data for cluster size Nc={Nc}.")
            for V in v_list:
                series = cluster_by_v.get(str(V)) or cluster_by_v.get(V)
                if series is None:
                    raise ValueError(f"Results payload missing rho_Q data for V={V} at Nc={Nc}.")
                arr = np.asarray(series, dtype=float)
                if arr.size != len(u_list):
                    raise ValueError(f"Cluster rho_Q series for Nc={Nc}, V={V} has length {arr.size}, expected {len(u_list)}.")
                cluster_rho_q[(Nc, V)] = arr

        stored_ratio_map = results.get('int_sep_ratios')
        if stored_ratio_map:
            converted_ratio_map = {}
            for key, val in stored_ratio_map.items():
                try:
                    Nc_key = int(key)
                except (TypeError, ValueError):
                    Nc_key = key
                converted_ratio_map[Nc_key] = _coerce_ratio(val, f"int_sep (Nc={Nc_key})")
            ratio_map = converted_ratio_map

        params_v_sep = params.get('v_sep_ratio', v_sep_ratio)
        v_sep_ratio = _coerce_ratio(params_v_sep, "V separation")
        t = params.get('t', t)
        L = params.get('L', L)
        chi = params.get('chi', chi)
        states_retained = params.get('states_retained', states_retained)

        mu_cache = np.asarray(results.get('mu_values', []), dtype=float)
        if mu_cache.size == 0 or mu_cache.shape != (len(u_list), len(v_list)):
            mu_cache = np.full((len(u_list), len(v_list)), np.nan, dtype=float)
    else:
        cluster_rho_q = {
            (Nc, V): np.full(len(u_list), np.nan, dtype=float)
            for Nc in cluster_sizes
            for V in v_list
        }
        finite_dmrg_rho_q = np.full((len(u_list), len(v_list)), np.nan, dtype=float)
        mu_cache = np.full((len(u_list), len(v_list)), np.nan, dtype=float)

        if set_filling is None:
            for u_idx, U in enumerate(u_list):
                mu0_guess = U / 2.0
                for v_idx in range(len(v_list)):
                    mu_cache[u_idx, v_idx] = mu0_guess
        else:
            print("=" * 60)
            print(f"Solving mu from cluster filling (Nc={mu_source_Nc}, n_target={set_filling})")
            print("=" * 60)

            int_sep_ratio = ratio_map[mu_source_Nc]
            for v_idx, V in enumerate(v_list):
                for u_idx, U in enumerate(u_list):
                    mu0_guess = U / 2.0
                    physical_params = PhysicalParams(U=U, mu_0=mu0_guess, V=V, t=t)
                    run_config = ClusterModelConfig(
                        L=L,
                        int_cluster_size=mu_source_Nc,
                        cluster_separation_ratio=int_sep_ratio,
                        V_separation_ratio=v_sep_ratio,
                        ham_lib='quspin',
                        physical_params=physical_params,
                        model_bc='periodic',
                        int_cluster_bc='periodic',
                        super_cluster_bc='periodic',
                        solver_method=solver_method,
                        states_retained=states_retained,
                    )
                    try:
                        _, _, mu_eff_avg = get_general_expectations(
                            run_config,
                            set_filling=set_filling,
                            return_mu=True,
                        )
                        if mu_eff_avg is None or not np.isfinite(mu_eff_avg):
                            raise ValueError(f"Invalid mu from cluster filling: {mu_eff_avg}")
                        mu_cache[u_idx, v_idx] = float(mu_eff_avg)
                    except Exception as exc:
                        failed_calculations.append({
                            'method': f'cluster mu (Nc={mu_source_Nc})',
                            'params': {'U': U, 'V': V},
                            'error': str(exc),
                        })

        print("\n")
        print("=" * 60)
        print("Running cluster rho_Q calculations for each N_c")
        print("=" * 60)
        for Nc in tqdm(cluster_sizes, desc="Cluster sizes", ncols=80):
            int_sep_ratio = ratio_map[Nc]
            for v_idx, V in enumerate(v_list):
                for u_idx, U in enumerate(u_list):
                    mu0_guess = U / 2.0
                    physical_params = PhysicalParams(U=U, mu_0=mu0_guess, V=V, t=t)
                    run_config = ClusterModelConfig(
                        L=L,
                        int_cluster_size=Nc,
                        cluster_separation_ratio=int_sep_ratio,
                        V_separation_ratio=v_sep_ratio,
                        ham_lib='quspin',
                        physical_params=physical_params,
                        model_bc='periodic',
                        int_cluster_bc='periodic',
                        super_cluster_bc='periodic',
                        solver_method=solver_method,
                        states_retained=states_retained,
                    )
                    try:
                        if set_filling is None:
                            rho_q = get_general_density_wave_observable(
                                run_config,
                                temperature=1e-2,
                                mu_eff=float(mu_cache[u_idx, v_idx]),
                            )
                        else:
                            rho_q = get_general_density_wave_observable(
                                run_config,
                                temperature=1e-2,
                                set_filling=set_filling,
                            )
                        cluster_rho_q[(Nc, V)][u_idx] = rho_q
                    except Exception as exc:
                        failed_calculations.append({
                            'method': f'cluster rho_Q (Nc={Nc})',
                            'params': {'U': U, 'V': V},
                            'error': str(exc),
                        })

        print("\n")
        print("=" * 60)
        print("Computing finite-DMRG rho_Q reference")
        print("=" * 60)
        for u_idx, U in enumerate(u_list):
            for v_idx, V in enumerate(v_list):
                mu_eff_value = float(mu_cache[u_idx, v_idx])
                try:
                    if dmrg_fixed_filling:
                        rho_q_dmrg = get_finite_dmrg_density_wave_observable(
                            L,
                            chi,
                            U,
                            t,
                            0.0,
                            V,
                            v_sep_ratio,
                            filling_target=set_filling,
                            dmrg_fixed_filling=True,
                        )
                    else:
                        if not np.isfinite(mu_eff_value):
                            raise ValueError("Skipping finite DMRG: mu was not determined from cluster data.")
                        rho_q_dmrg = get_finite_dmrg_density_wave_observable(
                            L,
                            chi,
                            U,
                            t,
                            mu_eff_value,
                            V,
                            v_sep_ratio,
                        )
                    finite_dmrg_rho_q[u_idx, v_idx] = rho_q_dmrg
                except Exception as exc:
                    failed_calculations.append({
                        'method': 'Finite DMRG rho_Q',
                        'params': {'U': U, 'V': V},
                        'error': str(exc),
                    })

    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf',
    ]
    color_map = {Nc: color_palette[idx % len(color_palette)] for idx, Nc in enumerate(cluster_sizes)}

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    figures, html_paths = create_u_value_rho_q_plots(
        u_list=u_list,
        v_list=v_list,
        cluster_sizes=cluster_sizes,
        cluster_rho_q=cluster_rho_q,
        finite_dmrg_rho_q=finite_dmrg_rho_q,
        v_sep_ratio=v_sep_ratio,
        t=t,
        L=L,
        chi=chi,
        states_retained=states_retained,
        color_map=color_map,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        timestamp=timestamp,
        show_plots=show_plots,
        save_html=save_html,
        set_filling=set_filling,
        mu_source_Nc=mu_source_Nc,
        dmrg_fixed_filling=dmrg_fixed_filling,
    )
    fig = figures[0] if figures else None
    saved_paths = {}
    if save_html:
        saved_paths['html_pages'] = html_paths
        if html_paths:
            saved_paths['html'] = html_paths[0]

    cluster_rho_q_serialized = {
        Nc: {V: cluster_rho_q[(Nc, V)].tolist() for V in v_list}
        for Nc in cluster_sizes
    }
    results_payload = {
        'cluster_sizes': cluster_sizes,
        'U_values': u_list,
        'V_values': v_list,
        'cluster_rho_q': cluster_rho_q_serialized,
        'finite_dmrg_rho_q': finite_dmrg_rho_q.tolist(),
        'mu_values': mu_cache.tolist(),
        'int_sep_ratios': {Nc: ratio_map[Nc] for Nc in cluster_sizes},
        'parameters': {
            'observable': 'rho_Q',
            'v_sep_ratio': v_sep_ratio,
            't': t,
            'L': L,
            'chi': chi,
            'solver_method': solver_method,
            'states_retained': states_retained,
            'set_filling': set_filling,
            'dmrg_fixed_filling': dmrg_fixed_filling,
            'finite_dmrg_only': True,
        },
        'artifacts': saved_paths,
        'failures': failed_calculations,
    }

    if save_data:
        pickle_name = f"{filename_prefix}_L{L}_chi{chi}_{timestamp}.pkl"
        pickle_path = os.path.join(output_dir, pickle_name)
        with open(pickle_path, 'wb') as fh:
            pickle.dump(results_payload, fh)
        saved_paths['pickle'] = pickle_path
        print(f"Saved data to {pickle_path}")

    if failed_calculations:
        print("\n" + "=" * 60)
        print("WARNING: Some calculations failed")
        print("=" * 60)
        for failure in failed_calculations:
            params_desc = ', '.join(f"{k}={v}" for k, v in failure['params'].items())
            print(f"{failure['method']}: {params_desc}")
            print(f"  Error: {failure['error'].splitlines()[0]}")

    return fig, results_payload


def compare_U_values_with_dmrg(
    v_sep_ratio: Tuple[int, int],
    int_sep_ratios: Union[Tuple[int, int], Dict[int, Tuple[int, int]]],
    cluster_sizes: Sequence[int],
    U_values: Sequence[float],
    V_values: Sequence[float],
    *,
    t: float = 1.0,
    L: int = 20,
    chi: int = 32,
    solver_method: str = 'dense_ED',
    states_retained: int = 4,
    output_dir: str = 'large_files/plots',
    show_plots: bool = True,
    save_html: bool = True,
    save_data: bool = True,
    filename_prefix: str = 'U_value_energy_comparison',
    log_yaxis: bool = True,
    include_idmrg: bool = True,
    include_finite_dmrg: bool = True,
    include_timing: bool = False,
    include_timing_plot: bool = False,
    plot_relative_error: bool = False,
    set_filling: Optional[float] = None,
    dmrg_fixed_filling: bool = False,
    results: Optional[Union[Dict, str, os.PathLike]] = None,
) -> Tuple[go.Figure, Dict]:
    """
    Plot how the cluster method converges to DMRG as the cluster size (N_c) increases.
    
    This function is similar to compare_cluster_sizes_with_dmrg but swaps the axes:
    - Subplots: Different V values
    - X-axis: U values
    - Lines: Different cluster sizes (N_c)
    - Y-axis: Ground state energy per site

    Args:
        v_sep_ratio: Ratio controlling the AA modulation for V.
        int_sep_ratios: Either a single (p, q) tuple applied to every cluster
            size or a dict mapping each N_c to its specific interaction
            separation ratio.
        cluster_sizes: Iterable of cluster sizes (N_c) to test.
        U_values: Iterable of U values; each becomes a point on the x-axis.
        V_values: Iterable of V strengths; each becomes a separate subplot.
        t: Hopping parameter.
        L: System size used for the cluster method.
        chi: Bond dimension for iDMRG.
        solver_method: Diagonalisation backend for the cluster Hamiltonian.
        states_retained: Number of states retained in the cluster solver.
        output_dir: Directory for saved artifacts.
        show_plots: Whether to open the generated Plotly figure.
        save_html: If True, save the interactive figure as HTML.
        save_data: If True, pickle the raw numerical results.
        filename_prefix: Prefix for saved artifact names.
        log_yaxis: Plot the y-axis on a log scale (not used for energy plots).
        include_idmrg: Whether to include iDMRG reference calculations.
        include_finite_dmrg: Whether to include finite DMRG reference calculations.
        plot_relative_error: If True, energy panels show |E_finite - E| / |E_finite| (finite DMRG as reference) with percent tick labels.
        set_filling: If provided, first solve for a chemical potential (from the cluster spectra)
            that targets this filling (per site), then use the resulting average mu in DMRG.
        dmrg_fixed_filling: If True, finite DMRG is run in the canonical ensemble (fixed N corresponding
            to set_filling) with mu=0. iDMRG remains grand-canonical using the cluster-derived mu.

    Returns:
        (figure, results_dict)
    """

    if t is None:
        raise ValueError("Parameter t must be specified for the cluster calculations.")
    if dmrg_fixed_filling and include_finite_dmrg and set_filling is None:
        raise ValueError("dmrg_fixed_filling=True requires set_filling to be provided.")

    def _coerce_ratio(value, label: str) -> Tuple[int, int]:
        if value is None:
            raise ValueError(f"{label} ratio must be provided.")
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{label} ratio must be a length-2 iterable, got {value!r}.")
        try:
            p = int(round(value[0]))
            q = int(round(value[1]))
        except Exception as exc:
            raise ValueError(f"Could not parse {label} ratio {value!r} into integers.") from exc
        if q == 0:
            raise ValueError(f"Denominator for {label} ratio cannot be zero.")
        return (p, q)

    cluster_sizes = sorted({int(size) for size in cluster_sizes})
    if not cluster_sizes:
        raise ValueError("Provide at least one cluster size (N_c).")

    U_values = np.asarray(U_values, dtype=float)
    V_values = np.asarray(V_values, dtype=float)
    if U_values.ndim != 1 or U_values.size == 0:
        raise ValueError("U_values must be a 1-D array with at least one entry.")
    if V_values.ndim != 1 or V_values.size == 0:
        raise ValueError("V_values must be a 1-D array with at least one entry.")

    u_list = [float(u) for u in U_values]
    v_list = [float(v) for v in V_values]

    v_sep_ratio = _coerce_ratio(v_sep_ratio, "V separation")

    if isinstance(int_sep_ratios, dict):
        ratio_map: Dict[int, Tuple[int, int]] = {}
        for Nc in cluster_sizes:
            if Nc not in int_sep_ratios:
                raise ValueError(f"No int_sep ratio provided for cluster size Nc={Nc}.")
            ratio_map[Nc] = _coerce_ratio(int_sep_ratios[Nc], f"int_sep (Nc={Nc})")
    else:
        common_ratio = _coerce_ratio(int_sep_ratios, "int_sep")
        ratio_map = {Nc: common_ratio for Nc in cluster_sizes}

    cluster_results: Dict[Tuple[int, float], np.ndarray]
    cluster_fillings: Dict[Tuple[int, float], np.ndarray]
    idmrg_cache: np.ndarray
    finite_dmrg_cache: np.ndarray
    idmrg_fill_cache: np.ndarray
    finite_dmrg_fill_cache: np.ndarray
    failed_calculations: List[Dict] = []
    timing_csv_path = os.environ.get("TIMING_CSV") if include_timing else None
    timing_recorder = TimingRecorder(csv_path=timing_csv_path) if include_timing else None

    if results is not None:
        if isinstance(results, (str, os.PathLike)):
            results_path = Path(results)
            if not results_path.exists():
                raise ValueError(f"Results file not found: {results_path}")
            with open(results_path, 'rb') as fh:
                results = pickle.load(fh)
        elif not isinstance(results, dict):
            raise ValueError("results must be a dict or path-like object when provided.")

        save_data = False  # Avoid re-saving when plotting from cached data
        print("Using precomputed results payload; skipping new simulations.")
        params = results.get('parameters', {})
        stored_set_filling = params.get('set_filling', set_filling)
        set_filling = float(stored_set_filling) if stored_set_filling is not None else None
        dmrg_fixed_filling = bool(params.get('dmrg_fixed_filling', dmrg_fixed_filling))
        
        # Load caches if available, otherwise fill with NaN
        idmrg_cache = np.asarray(results.get('idmrg_energies', []), dtype=float)
        if idmrg_cache.size == 0:
             idmrg_cache = np.full((len(u_list), len(v_list)), np.nan, dtype=float)
             
        finite_dmrg_cache = np.asarray(results.get('finite_dmrg_energies', []), dtype=float)
        if finite_dmrg_cache.size == 0:
             finite_dmrg_cache = np.full((len(u_list), len(v_list)), np.nan, dtype=float)
        idmrg_fill_cache = np.asarray(results.get('idmrg_fillings', []), dtype=float)
        if idmrg_fill_cache.size == 0:
             idmrg_fill_cache = np.full((len(u_list), len(v_list)), np.nan, dtype=float)
        finite_dmrg_fill_cache = np.asarray(results.get('finite_dmrg_fillings', []), dtype=float)
        if finite_dmrg_fill_cache.size == 0:
             finite_dmrg_fill_cache = np.full((len(u_list), len(v_list)), np.nan, dtype=float)

        serialized_clusters = results.get('cluster_energies', {})
        cluster_results = {}
        cluster_fillings = {}

        for Nc in cluster_sizes:
            cluster_by_v = serialized_clusters.get(str(Nc)) or serialized_clusters.get(Nc)
            if cluster_by_v is None:
                raise ValueError(f"Results payload missing data for cluster size Nc={Nc}.")
            for V in v_list:
                series = cluster_by_v.get(str(V)) or cluster_by_v.get(V)
                if series is None:
                    raise ValueError(f"Results payload missing data for V={V} at Nc={Nc}.")
                arr = np.asarray(series, dtype=float)
                if arr.size != len(u_list):
                    raise ValueError(f"Cluster series for Nc={Nc}, V={V} has length {arr.size}, expected {len(u_list)}.")
                cluster_results[(Nc, V)] = arr
        # Optional: load fillings if present
        serialized_fills = results.get('cluster_fillings', {})
        for Nc in cluster_sizes:
            cluster_by_v = serialized_fills.get(str(Nc)) or serialized_fills.get(Nc) or {}
            for V in v_list:
                series = cluster_by_v.get(str(V)) or cluster_by_v.get(V)
                if series is None:
                    cluster_fillings[(Nc, V)] = np.full(len(u_list), np.nan, dtype=float)
                    continue
                arr = np.asarray(series, dtype=float)
                if arr.size != len(u_list):
                    cluster_fillings[(Nc, V)] = np.full(len(u_list), np.nan, dtype=float)
                else:
                    cluster_fillings[(Nc, V)] = arr

        # Override metadata from results where available
        stored_ratio_map = results.get('int_sep_ratios')
        if stored_ratio_map:
            converted_ratio_map = {}
            for key, val in stored_ratio_map.items():
                try:
                    Nc_key = int(key)
                except (TypeError, ValueError):
                    Nc_key = key
                converted_ratio_map[Nc_key] = _coerce_ratio(val, f"int_sep (Nc={Nc_key})")
            ratio_map = converted_ratio_map
        params_v_sep = results.get('parameters', {}).get('v_sep_ratio', v_sep_ratio)
        v_sep_ratio = _coerce_ratio(params_v_sep, "V separation")
        t = params.get('t', t)
        L = params.get('L', L)
        chi = params.get('chi', chi)
        states_retained = params.get('states_retained', states_retained)
        include_idmrg = params.get('include_idmrg', include_idmrg)
        include_finite_dmrg = params.get('include_finite_dmrg', include_finite_dmrg)
        if dmrg_fixed_filling and include_finite_dmrg and set_filling is None:
            raise ValueError("Cached results have dmrg_fixed_filling=True but set_filling is missing.")
        if plot_relative_error and not include_finite_dmrg:
            raise ValueError("Cached results do not include finite DMRG data required for plot_relative_error.")
    else:
        # Storage for computed energies
        cluster_results = {
            (Nc, V): np.full(len(u_list), np.nan, dtype=float)
            for Nc in cluster_sizes
            for V in v_list
        }
        cluster_fillings = {
            (Nc, V): np.full(len(u_list), np.nan, dtype=float)
            for Nc in cluster_sizes
            for V in v_list
        }
        idmrg_cache = np.full((len(u_list), len(v_list)), np.nan, dtype=float)
        finite_dmrg_cache = np.full((len(u_list), len(v_list)), np.nan, dtype=float)
        idmrg_fill_cache = np.full((len(u_list), len(v_list)), np.nan, dtype=float)
        finite_dmrg_fill_cache = np.full((len(u_list), len(v_list)), np.nan, dtype=float)

        mu_cache = np.full((len(u_list), len(v_list)), np.nan, dtype=float)
        mu_source_Nc = max(cluster_sizes)

        if set_filling is None:
            for u_idx, U in enumerate(u_list):
                mu0_guess = U / 2.0
                for v_idx in range(len(v_list)):
                    mu_cache[u_idx, v_idx] = mu0_guess
        else:
            print("=" * 60)
            print(f"Solving mu from cluster filling (Nc={mu_source_Nc}, n_target={set_filling})")
            print("=" * 60)

            int_sep_ratio = ratio_map[mu_source_Nc]
            super_cluster_size = None
            if timing_recorder is not None:
                try:
                    clusters_tmp = generate_clusters(L, mu_source_Nc, int_sep_ratio, v_sep_ratio)
                    super_cluster_size = supercluster_size_from_clusters(clusters_tmp)
                except Exception:
                    super_cluster_size = None

            for v_idx, V in enumerate(v_list):
                for u_idx, U in enumerate(u_list):
                    mu0_guess = U / 2.0
                    physical_params = PhysicalParams(U=U, mu_0=mu0_guess, V=V, t=t)
                    run_config = ClusterModelConfig(
                        L=L,
                        int_cluster_size=mu_source_Nc,
                        cluster_separation_ratio=int_sep_ratio,
                        V_separation_ratio=v_sep_ratio,
                        ham_lib='quspin',
                        physical_params=physical_params,
                        model_bc='periodic',
                        int_cluster_bc='periodic',
                        super_cluster_bc='periodic',
                        solver_method=solver_method,
                        states_retained=states_retained,
                    )
                    try:
                        meta = {
                            "method": "cluster_mu",
                            "U": U,
                            "V": V,
                            "t": t,
                            "L": L,
                            "Nc": mu_source_Nc,
                            "int_sep": int_sep_ratio,
                            "v_sep": v_sep_ratio,
                            "super_cluster_size": super_cluster_size,
                        }
                        system_expectations, _, mu_eff_avg = time_call(
                            timing_recorder,
                            meta,
                            get_general_expectations,
                            run_config,
                            timing_recorder=timing_recorder,
                            set_filling=set_filling,
                            return_mu=True,
                        )
                        if mu_eff_avg is None or not np.isfinite(mu_eff_avg):
                            raise ValueError(f"Invalid mu from cluster filling: {mu_eff_avg}")
                        mu_cache[u_idx, v_idx] = float(mu_eff_avg)

                        energy, filling, _ = system_expectations
                        energy_subtracted = (energy + mu0_guess * filling) / L
                        cluster_results[(mu_source_Nc, V)][u_idx] = energy_subtracted
                        cluster_fillings[(mu_source_Nc, V)][u_idx] = filling / L
                    except Exception as exc:
                        import traceback
                        error_msg = str(exc) or f"{type(exc).__name__}: {repr(exc)}"
                        failed_calculations.append({
                            'method': f'cluster mu (Nc={mu_source_Nc})',
                            'params': {'U': U, 'V': V},
                            'error': error_msg,
                            'traceback': traceback.format_exc(),
                        })

        print("\n")
        print("=" * 60)
        print("Running cluster calculations for each N_c")
        print("=" * 60)
        for Nc in tqdm(cluster_sizes, desc="Cluster sizes", ncols=80):
            if set_filling is not None and Nc == mu_source_Nc:
                continue
            int_sep_ratio = ratio_map[Nc]
            super_cluster_size = None
            if timing_recorder is not None:
                try:
                    clusters_tmp = generate_clusters(L, Nc, int_sep_ratio, v_sep_ratio)
                    super_cluster_size = supercluster_size_from_clusters(clusters_tmp)
                except Exception:
                    super_cluster_size = None
            for v_idx, V in enumerate(v_list):
                for u_idx, U in enumerate(u_list):
                    mu0_guess = U / 2.0
                    physical_params = PhysicalParams(U=U, mu_0=mu0_guess, V=V, t=t)
                    run_config = ClusterModelConfig(
                        L=L,
                        int_cluster_size=Nc,
                        cluster_separation_ratio=int_sep_ratio,
                        V_separation_ratio=v_sep_ratio,
                        ham_lib='quspin',
                        physical_params=physical_params,
                        model_bc='periodic',
                        int_cluster_bc='periodic',
                        super_cluster_bc='periodic',
                        solver_method=solver_method,
                        states_retained=states_retained,
                    )
                    try:
                        meta = {
                            "method": "cluster_ED",
                            "U": U,
                            "V": V,
                            "t": t,
                            "L": L,
                            "Nc": Nc,
                            "int_sep": int_sep_ratio,
                            "v_sep": v_sep_ratio,
                            "super_cluster_size": super_cluster_size,
                        }
                        if set_filling is None:
                            system_expectations, _ = time_call(
                                timing_recorder,
                                meta,
                                get_general_expectations,
                                run_config,
                                timing_recorder=timing_recorder,
                                mu_eff=float(mu_cache[u_idx, v_idx]),
                            )
                        else:
                            system_expectations, _ = time_call(
                                timing_recorder,
                                meta,
                                get_general_expectations,
                                run_config,
                                timing_recorder=timing_recorder,
                                set_filling=set_filling,
                            )
                        energy, filling, _ = system_expectations
                        energy_subtracted = (energy + mu0_guess * filling) / L
                        cluster_results[(Nc, V)][u_idx] = energy_subtracted
                        cluster_fillings[(Nc, V)][u_idx] = filling / L
                    except Exception as exc:
                        import traceback
                        error_msg = str(exc) or f"{type(exc).__name__}: {repr(exc)}"
                        failed_calculations.append({
                            'method': f'cluster Nc={Nc}',
                            'params': {'U': U, 'V': V},
                            'error': error_msg,
                            'traceback': traceback.format_exc(),
                        })

        print("\n")
        print("=" * 60)
        print("Computing reference energies")
        print("=" * 60)

        for u_idx, U in enumerate(u_list):
            for v_idx, V in enumerate(v_list):
                mu_eff_value = float(mu_cache[u_idx, v_idx])

                # iDMRG
                if include_idmrg:
                    try:
                        if not np.isfinite(mu_eff_value):
                            raise ValueError("Skipping iDMRG: mu was not determined from cluster data.")
                        meta = {
                            "method": "iDMRG",
                            "U": U,
                            "V": V,
                            "t": t,
                            "L": L,
                            "Nc": None,
                            "int_sep": None,
                            "v_sep": v_sep_ratio,
                            "super_cluster_size": None,
                        }
                        energy_dmrg, filling_dmrg, _ = time_call(
                            timing_recorder,
                            meta,
                            run_dmrg_method,
                            U,
                            mu_eff_value,
                            V,
                            v_sep_ratio,
                            t,
                            L,
                            chi,
                        )
                        dmrg_value = energy_dmrg + mu_eff_value * filling_dmrg
                        idmrg_cache[u_idx, v_idx] = dmrg_value
                        idmrg_fill_cache[u_idx, v_idx] = filling_dmrg
                    except Exception as exc:
                        import traceback
                        error_msg = str(exc) or f"{type(exc).__name__}: {repr(exc)}"
                        failed_calculations.append({
                            'method': 'iDMRG',
                            'params': {'U': U, 'V': V},
                            'error': error_msg,
                            'traceback': traceback.format_exc(),
                        })

                # Finite DMRG
                if include_finite_dmrg:
                    try:
                        meta = {
                            "method": "DMRG",
                            "U": U,
                            "V": V,
                            "t": t,
                            "L": L,
                            "Nc": None,
                            "int_sep": None,
                            "v_sep": v_sep_ratio,
                            "super_cluster_size": None,
                        }
                        if dmrg_fixed_filling:
                            energy_finite, _, filling_finite = time_call(
                                timing_recorder,
                                meta,
                                get_gnd_fixed_filling,
                                L,
                                chi,
                                set_filling,
                                U,
                                t,
                                V,
                                v_sep_ratio,
                            )
                            energy_finite_per_site = energy_finite / L
                            dmrg_value = energy_finite_per_site
                        else:
                            if not np.isfinite(mu_eff_value):
                                raise ValueError("Skipping finite DMRG: mu was not determined from cluster data.")
                            energy_finite, _, filling_finite = time_call(
                                timing_recorder,
                                meta,
                                get_gnd,
                                L,
                                chi,
                                U,
                                t,
                                mu_eff_value,
                                V,
                                v_sep_ratio,
                            )
                            energy_finite_per_site = energy_finite / L
                            dmrg_value = energy_finite_per_site + mu_eff_value * filling_finite
                        finite_dmrg_cache[u_idx, v_idx] = dmrg_value
                        finite_dmrg_fill_cache[u_idx, v_idx] = filling_finite
                    except Exception as exc:
                        import traceback
                        error_msg = str(exc) or f"{type(exc).__name__}: {repr(exc)}"
                        failed_calculations.append({
                            'method': 'Finite DMRG',
                            'params': {'U': U, 'V': V},
                            'error': error_msg,
                            'traceback': traceback.format_exc(),
                        })

    # --- PLOTTING LOGIC ---
    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf',
    ]
    color_map = {Nc: color_palette[idx % len(color_palette)] for idx, Nc in enumerate(cluster_sizes)}

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    figures, html_paths = create_u_value_comparison_plots(
        u_list=u_list,
        v_list=v_list,
        cluster_sizes=cluster_sizes,
        cluster_results=cluster_results,
        cluster_fillings=cluster_fillings,
        idmrg_cache=idmrg_cache,
        finite_dmrg_cache=finite_dmrg_cache,
        idmrg_fill_cache=idmrg_fill_cache,
        finite_dmrg_fill_cache=finite_dmrg_fill_cache,
        v_sep_ratio=v_sep_ratio,
        t=t,
        L=L,
        chi=chi,
        states_retained=states_retained,
        color_map=color_map,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        timestamp=timestamp,
        show_plots=show_plots,
        save_html=save_html,
        set_filling=set_filling,
        mu_source_Nc=max(cluster_sizes) if set_filling is not None else None,
        dmrg_fixed_filling=dmrg_fixed_filling,
        plot_relative_error=plot_relative_error,
    )
    fig = figures[0] if figures else None
    saved_paths = {}
    if save_html:
        saved_paths['html_pages'] = html_paths
        if html_paths:
            saved_paths['html'] = html_paths[0]

    cluster_energy_serialized = {
        Nc: {V: cluster_results[(Nc, V)].tolist() for V in v_list}
        for Nc in cluster_sizes
    }
    cluster_fillings_serialized = {
        Nc: {V: cluster_fillings[(Nc, V)].tolist() for V in v_list}
        for Nc in cluster_sizes
    }
    mu_values_serialized = mu_cache.tolist() if 'mu_cache' in locals() else None
    results_payload = {
        'cluster_sizes': cluster_sizes,
        'U_values': u_list,
        'V_values': v_list,
        'cluster_energies': cluster_energy_serialized,
        'cluster_fillings': cluster_fillings_serialized,
        'idmrg_energies': idmrg_cache.tolist(),
        'finite_dmrg_energies': finite_dmrg_cache.tolist(),
        'idmrg_fillings': idmrg_fill_cache.tolist(),
        'finite_dmrg_fillings': finite_dmrg_fill_cache.tolist(),
        'mu_values': mu_values_serialized,
        'int_sep_ratios': {Nc: ratio_map[Nc] for Nc in cluster_sizes},
        'parameters': {
            'v_sep_ratio': v_sep_ratio,
            't': t,
            'L': L,
            'chi': chi,
            'solver_method': solver_method,
            'states_retained': states_retained,
            'include_idmrg': include_idmrg,
            'include_finite_dmrg': include_finite_dmrg,
            'plot_relative_error': plot_relative_error,
            'set_filling': set_filling,
            'dmrg_fixed_filling': dmrg_fixed_filling,
        },
        'artifacts': saved_paths,
        'failures': failed_calculations,
    }
    if include_timing and timing_recorder is not None:
        results_payload['timings'] = timing_recorder.records
        if include_timing_plot and timing_recorder.records:
            fig_timing, timing_artifacts = plot_timings(
                timing_recorder.records,
                output_dir=output_dir,
                filename_prefix=f"{filename_prefix}_timing",
                show_plots=show_plots,
            )
            saved_paths['timing_plot'] = timing_artifacts.get('html')

    if save_data:
        pickle_name = f"{filename_prefix}_L{L}_chi{chi}_{timestamp}.pkl"
        pickle_path = os.path.join(output_dir, pickle_name)
        with open(pickle_path, 'wb') as fh:
            pickle.dump(results_payload, fh)
        saved_paths['pickle'] = pickle_path
        print(f"Saved data to {pickle_path}")

    if failed_calculations:
        print("\n" + "=" * 60)
        print("WARNING: Some calculations failed")
        print("=" * 60)
        for failure in failed_calculations:
            params_desc = ', '.join(f"{k}={v}" for k, v in failure['params'].items())
            print(f"{failure['method']}: {params_desc}")
            error_lines = failure['error'].splitlines()
            print(f"  Error: {error_lines[0] if error_lines else '(no error message)'}")
            if 'traceback' in failure:
                print(f"  Full traceback:\n{failure['traceback']}")

    if show_plots:
        fig.show()

    return fig, results_payload


def compare_filling_with_int_cluster_sizes(
    int_sep_ratios_by_Nc: Dict[int, Sequence[Tuple[int, int]]],
    U_values: Sequence[float],
    *,
    V: float = 0.0,
    v_sep_ratio: Optional[Tuple[int, int]] = None,
    t: float = 1.0,
    L: int = 20,
    chi: int = 32,
    solver_method: str = 'dense_ED',
    states_retained: int = 4,
    output_dir: str = 'large_files/plots',
    show_plots: bool = True,
    save_html: bool = True,
    save_data: bool = True,
    filename_prefix: str = 'filling_int_cluster_comparison',
    include_idmrg: bool = True,
    include_finite_dmrg: bool = True,
    include_timing: bool = False,
    include_timing_plot: bool = False,
    plot_relative_error: bool = False,
    set_filling: Optional[float] = None,
    dmrg_fixed_filling: bool = True,
    fixed_mu: Optional[Union[float, str]] = None,
    compute_localization: bool = False,
    results: Optional[Union[Dict, str, os.PathLike]] = None,
    cols_per_page: int = 3,
) -> Tuple[go.Figure, Dict]:
    """
    Compare half-filling and quarter-filling results across cluster sizes and interaction separations.

    Creates a 2xN grid per page where:
    - Top row: Half-filling (n=1 per site) for each cluster size
    - Bottom row: Quarter-filling (n=0.5 per site) for the same cluster sizes
    - X-axis: U values
    - Lines on each subplot: Different interaction cluster separations compatible with that N_c
    - Y-axis: Ground state energy per site (or relative error if plot_relative_error=True)

    When V≈0 (|V| < 1e-6), automatically sets v_sep_ratio to (1,1) to avoid cluster fusion,
    which produces an inert onsite term instead of inter-cluster hopping.

    Args:
        int_sep_ratios_by_Nc: Dict mapping each cluster size (N_c) to a list of compatible
            interaction separation ratios. E.g.:
            {
                2: [(1, 2), (1, 4), (1, 8)],   # π, π/2, π/4 for N_c=2
                3: [(1, 3), (2, 3)],            # π/3, 2π/3 for N_c=3
                4: [(1, 4), (1, 8)],            # π/2, π/4 for N_c=4
            }
            The keys determine which cluster sizes are plotted (one column per N_c).
        U_values: U values for x-axis.
        V: Fixed V value across all subplots (default 0).
        v_sep_ratio: Ratio controlling the AA modulation for V. If None and |V| < 1e-6,
            defaults to (1,1) to avoid fusion. Otherwise defaults to (1,2).
        t: Hopping parameter.
        L: System size for the cluster method.
        chi: Bond dimension for iDMRG.
        solver_method: Diagonalisation backend for the cluster Hamiltonian.
        states_retained: Number of states retained in the cluster solver.
        output_dir: Directory for saved artifacts.
        show_plots: Whether to open the generated Plotly figure.
        save_html: If True, save the interactive figure as HTML.
        save_data: If True, pickle the raw numerical results.
        filename_prefix: Prefix for saved artifact names.
        include_idmrg: Whether to include iDMRG reference calculations.
        include_finite_dmrg: Whether to include finite DMRG reference calculations.
        include_timing: Whether to record timing information.
        include_timing_plot: Whether to generate timing plots.
        plot_relative_error: If True, show |E_finite - E| / |E_finite| instead of energy.
        set_filling: If provided, overrides the default half/quarter filling logic.
            When None (default), top row targets n=1 (half-filling) and bottom row targets n=0.5.
        dmrg_fixed_filling: If True (default), finite DMRG uses canonical ensemble (fixed N)
            to match the target filling. Set to False for grand canonical DMRG.
        fixed_mu: Fix the chemical potential instead of targeting a specific filling.
            - None (default): target filling via bisection (current behavior)
            - "U/2": use μ = U/2 (particle-hole symmetric point)
            - "0": use μ = 0
            - "both": compare μ=U/2 (top row) vs μ=0 (bottom row)
            - float: use that literal μ value
            When set, filling becomes an output variable rather than a target.
        compute_localization: If True, compute and store correlation length from iDMRG.
            Only works when include_idmrg=True. Uses TeNPy's transfer matrix method.
        results: Pre-computed results dict or path to pickle file to load instead of computing.
        cols_per_page: Maximum number of subplot columns per page (default: 3).
            Additional columns overflow to new pages.

    Returns:
        (figure, results_dict)
    """
    # Warn if compute_localization is True but iDMRG is disabled
    if compute_localization and not include_idmrg:
        warnings.warn(
            "compute_localization=True but include_idmrg=False. "
            "Correlation length requires iDMRG data. No correlation lengths will be computed.",
            UserWarning,
        )
        compute_localization = False  # Disable to avoid downstream issues

    if t is None:
        raise ValueError("Parameter t must be specified for the cluster calculations.")

    # Handle V≈0 case: use inert v_sep_ratio to avoid cluster fusion
    V_ZERO_THRESHOLD = 1e-6
    if v_sep_ratio is None:
        if abs(V) < V_ZERO_THRESHOLD:
            v_sep_ratio = (1, 1)  # Step = L, creates onsite term, no fusion
            print(f"V≈0 detected: using v_sep_ratio=(1,1) to avoid cluster fusion")
        else:
            v_sep_ratio = (1, 2)  # Default π spacing

    def _coerce_ratio(value, label: str) -> Tuple[int, int]:
        if value is None:
            raise ValueError(f"{label} ratio must be provided.")
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{label} ratio must be a length-2 iterable, got {value!r}.")
        try:
            p = int(round(value[0]))
            q = int(round(value[1]))
        except Exception as exc:
            raise ValueError(f"Could not parse {label} ratio {value!r} into integers.") from exc
        if q == 0:
            raise ValueError(f"Denominator for {label} ratio cannot be zero.")
        return (p, q)

    # Parse and validate int_sep_ratios_by_Nc
    if not int_sep_ratios_by_Nc:
        raise ValueError("Provide at least one cluster size with int_sep_ratios.")

    # Build validated structure: {Nc: [coerced ratios]}
    int_sep_map: Dict[int, List[Tuple[int, int]]] = {}
    for Nc, ratios in int_sep_ratios_by_Nc.items():
        Nc = int(Nc)
        if not ratios:
            raise ValueError(f"No int_sep_ratios provided for Nc={Nc}.")
        int_sep_map[Nc] = [_coerce_ratio(r, f"int_sep (Nc={Nc})") for r in ratios]

    cluster_sizes = sorted(int_sep_map.keys())

    # Collect all unique int_sep_ratios across all Nc (for color mapping)
    all_int_sep_ratios: List[Tuple[int, int]] = []
    for ratios in int_sep_map.values():
        for r in ratios:
            if r not in all_int_sep_ratios:
                all_int_sep_ratios.append(r)

    U_values = np.asarray(U_values, dtype=float)
    if U_values.ndim != 1 or U_values.size == 0:
        raise ValueError("U_values must be a 1-D array with at least one entry.")

    u_list = [float(u) for u in U_values]
    v_sep_ratio = _coerce_ratio(v_sep_ratio, "V separation")

    # Define the two filling modes:
    # - Half-filling: n=1 per site (for spinful fermions, 1 up + 0 down or 0.5 up + 0.5 down on average)
    # - Quarter-filling: n=0.5 per site
    # We use set_filling to target these values
    filling_modes = ['half', 'quarter']  # top row, bottom row
    filling_targets = {'half': 1.0, 'quarter': 0.5}  # target filling per site

    # Handle fixed_mu mode: fix chemical potential instead of targeting filling
    use_fixed_mu = fixed_mu is not None
    if use_fixed_mu:
        if fixed_mu == "both":
            # Compare μ=U/2 (top row) vs μ=0 (bottom row)
            filling_modes = ['mu_half_U', 'mu_zero']
            # mu_values will be computed per-U in the loop
            mu_mode_funcs = {
                'mu_half_U': lambda U: U / 2.0,
                'mu_zero': lambda U: 0.0,
            }
            print("Fixed μ mode: comparing μ=U/2 (top row) vs μ=0 (bottom row)")
        elif fixed_mu == "U/2":
            filling_modes = ['mu_half_U']
            mu_mode_funcs = {'mu_half_U': lambda U: U / 2.0}
            print("Fixed μ mode: using μ=U/2")
        elif fixed_mu == "0" or (isinstance(fixed_mu, (int, float)) and fixed_mu == 0):
            filling_modes = ['mu_zero']
            mu_mode_funcs = {'mu_zero': lambda U: 0.0}
            print("Fixed μ mode: using μ=0")
        elif isinstance(fixed_mu, (int, float)):
            filling_modes = ['mu_fixed']
            mu_val = float(fixed_mu)
            mu_mode_funcs = {'mu_fixed': lambda U, _val=mu_val: _val}
            print(f"Fixed μ mode: using μ={mu_val}")
        else:
            raise ValueError(f"Invalid fixed_mu value: {fixed_mu!r}. "
                           "Use None, 'U/2', '0', 'both', or a numeric value.")

    # Storage structures
    # Key: (Nc, int_sep_ratio, filling_mode) -> array of energies indexed by U
    cluster_results: Dict[Tuple[int, Tuple[int, int], str], np.ndarray]
    cluster_fillings: Dict[Tuple[int, Tuple[int, int], str], np.ndarray]
    # DMRG caches: indexed by (u_idx, filling_mode_idx)
    idmrg_cache: np.ndarray
    finite_dmrg_cache: np.ndarray
    idmrg_fill_cache: np.ndarray
    finite_dmrg_fill_cache: np.ndarray
    failed_calculations: List[Dict] = []
    timing_csv_path = os.environ.get("TIMING_CSV") if include_timing else None
    timing_recorder = TimingRecorder(csv_path=timing_csv_path) if include_timing else None

    if results is not None:
        # Load from pre-computed results
        if isinstance(results, (str, os.PathLike)):
            results_path = Path(results)
            if not results_path.exists():
                raise ValueError(f"Results file not found: {results_path}")
            with open(results_path, 'rb') as fh:
                results = pickle.load(fh)
        elif not isinstance(results, dict):
            raise ValueError("results must be a dict or path-like object when provided.")

        save_data = False
        print("Using precomputed results payload; skipping new simulations.")
        params = results.get('parameters', {})

        # Load caches
        idmrg_cache = np.asarray(results.get('idmrg_energies', []), dtype=float)
        if idmrg_cache.size == 0:
            idmrg_cache = np.full((len(u_list), 2), np.nan, dtype=float)
        finite_dmrg_cache = np.asarray(results.get('finite_dmrg_energies', []), dtype=float)
        if finite_dmrg_cache.size == 0:
            finite_dmrg_cache = np.full((len(u_list), 2), np.nan, dtype=float)
        idmrg_fill_cache = np.asarray(results.get('idmrg_fillings', []), dtype=float)
        if idmrg_fill_cache.size == 0:
            idmrg_fill_cache = np.full((len(u_list), 2), np.nan, dtype=float)
        finite_dmrg_fill_cache = np.asarray(results.get('finite_dmrg_fillings', []), dtype=float)
        if finite_dmrg_fill_cache.size == 0:
            finite_dmrg_fill_cache = np.full((len(u_list), 2), np.nan, dtype=float)
        # Load correlation length if available
        if compute_localization:
            idmrg_corr_length_cache = np.asarray(results.get('idmrg_correlation_lengths', []), dtype=float)
            if idmrg_corr_length_cache.size == 0:
                idmrg_corr_length_cache = np.full((len(u_list), 2), np.nan, dtype=float)
        else:
            idmrg_corr_length_cache = None

        # Load cluster results
        serialized_clusters = results.get('cluster_energies', {})
        serialized_fills = results.get('cluster_fillings', {})
        cluster_results = {}
        cluster_fillings = {}

        for Nc in cluster_sizes:
            for int_sep in int_sep_map[Nc]:
                int_sep_key = f"{int_sep[0]}_{int_sep[1]}"
                for fill_idx, fill_mode in enumerate(filling_modes):
                    key = (Nc, int_sep, fill_mode)

                    cluster_by_nc = serialized_clusters.get(str(Nc), {})
                    cluster_by_sep = cluster_by_nc.get(int_sep_key, {})
                    series = cluster_by_sep.get(fill_mode)
                    if series is None:
                        cluster_results[key] = np.full(len(u_list), np.nan, dtype=float)
                    else:
                        cluster_results[key] = np.asarray(series, dtype=float)

                    fill_by_nc = serialized_fills.get(str(Nc), {})
                    fill_by_sep = fill_by_nc.get(int_sep_key, {})
                    fill_series = fill_by_sep.get(fill_mode)
                    if fill_series is None:
                        cluster_fillings[key] = np.full(len(u_list), np.nan, dtype=float)
                    else:
                        cluster_fillings[key] = np.asarray(fill_series, dtype=float)

        # Override metadata from results
        params_v_sep = params.get('v_sep_ratio', v_sep_ratio)
        v_sep_ratio = _coerce_ratio(params_v_sep, "V separation")
        t = params.get('t', t)
        L = params.get('L', L)
        chi = params.get('chi', chi)
        V = params.get('V', V)
        states_retained = params.get('states_retained', states_retained)
        include_idmrg = params.get('include_idmrg', include_idmrg)
        include_finite_dmrg = params.get('include_finite_dmrg', include_finite_dmrg)
    else:
        # Compute fresh results
        cluster_results = {
            (Nc, int_sep, fill_mode): np.full(len(u_list), np.nan, dtype=float)
            for Nc in cluster_sizes
            for int_sep in int_sep_map[Nc]
            for fill_mode in filling_modes
        }
        cluster_fillings = {
            (Nc, int_sep, fill_mode): np.full(len(u_list), np.nan, dtype=float)
            for Nc in cluster_sizes
            for int_sep in int_sep_map[Nc]
            for fill_mode in filling_modes
        }
        idmrg_cache = np.full((len(u_list), 2), np.nan, dtype=float)
        finite_dmrg_cache = np.full((len(u_list), 2), np.nan, dtype=float)
        idmrg_fill_cache = np.full((len(u_list), 2), np.nan, dtype=float)
        finite_dmrg_fill_cache = np.full((len(u_list), 2), np.nan, dtype=float)
        # Correlation length cache (only for iDMRG)
        idmrg_corr_length_cache = np.full((len(u_list), 2), np.nan, dtype=float) if compute_localization else None

        print("=" * 60)
        print("Running cluster calculations")
        print(f"Cluster sizes: {cluster_sizes}")
        for Nc in cluster_sizes:
            print(f"  Nc={Nc}: {[format_sep_as_pi(r) for r in int_sep_map[Nc]]}")
        print(f"V = {V}, v_sep = {format_sep_as_pi(v_sep_ratio)}")
        print("=" * 60)

        for Nc in tqdm(cluster_sizes, desc="Cluster sizes", ncols=80):
            for int_sep in int_sep_map[Nc]:
                for fill_idx, fill_mode in enumerate(filling_modes):
                    for u_idx, U in enumerate(u_list):
                        # Determine target filling or fixed mu based on mode
                        if use_fixed_mu:
                            # Fixed mu mode: use specified chemical potential
                            target_filling = None
                            mu0 = mu_mode_funcs[fill_mode](U)
                        elif set_filling is not None:
                            target_filling = set_filling
                            mu0 = U / 2.0  # Initial guess
                        else:
                            target_filling = filling_targets[fill_mode]
                            mu0 = U / 2.0  # Initial guess

                        physical_params = PhysicalParams(U=U, mu_0=mu0, V=V, t=t)
                        run_config = ClusterModelConfig(
                            L=L,
                            int_cluster_size=Nc,
                            cluster_separation_ratio=int_sep,
                            V_separation_ratio=v_sep_ratio,
                            ham_lib='quspin',
                            physical_params=physical_params,
                            model_bc='periodic',
                            int_cluster_bc='periodic',
                            super_cluster_bc='periodic',
                            solver_method=solver_method,
                            states_retained=states_retained,
                        )

                        try:
                            super_cluster_size = None
                            if timing_recorder is not None:
                                try:
                                    clusters_tmp = generate_clusters(L, Nc, int_sep, v_sep_ratio)
                                    super_cluster_size = supercluster_size_from_clusters(clusters_tmp)
                                except Exception:
                                    pass

                            meta = {
                                "method": "cluster_ED",
                                "U": U,
                                "V": V,
                                "t": t,
                                "L": L,
                                "Nc": Nc,
                                "int_sep": int_sep,
                                "v_sep": v_sep_ratio,
                                "super_cluster_size": super_cluster_size,
                                "filling_mode": fill_mode,
                                "target_filling": target_filling,
                                "fixed_mu": mu0 if use_fixed_mu else None,
                            }

                            if use_fixed_mu:
                                # Fixed mu mode: pass mu_eff directly, no filling target
                                system_expectations, _, _ = time_call(
                                    timing_recorder,
                                    meta,
                                    get_general_expectations,
                                    run_config,
                                    timing_recorder=timing_recorder,
                                    mu_eff=mu0,
                                    return_mu=True,
                                )
                            else:
                                # Target filling mode: use bisection to find mu
                                system_expectations, _, _ = time_call(
                                    timing_recorder,
                                    meta,
                                    get_general_expectations,
                                    run_config,
                                    timing_recorder=timing_recorder,
                                    set_filling=target_filling,
                                    return_mu=True,
                                )

                            energy, filling, _ = system_expectations
                            energy_subtracted = (energy + mu0 * filling) / L
                            key = (Nc, int_sep, fill_mode)
                            cluster_results[key][u_idx] = energy_subtracted
                            cluster_fillings[key][u_idx] = filling / L
                        except Exception as exc:
                            import traceback
                            error_msg = str(exc) or f"{type(exc).__name__}: {repr(exc)}"
                            failed_calculations.append({
                                'method': f'cluster Nc={Nc}, int_sep={format_sep_as_pi(int_sep)}, {fill_mode}',
                                'params': {'U': U, 'V': V},
                                'error': error_msg,
                                'traceback': traceback.format_exc(),
                            })

        # Compute DMRG references for each filling mode
        print("\n")
        print("=" * 60)
        print("Computing reference energies")
        print("=" * 60)

        for fill_idx, fill_mode in enumerate(filling_modes):
            for u_idx, U in enumerate(u_list):
                # Determine target filling or fixed mu for DMRG
                if use_fixed_mu:
                    target_fill = None
                    mu_eff_value = mu_mode_funcs[fill_mode](U)
                elif set_filling is not None:
                    target_fill = set_filling
                    mu_eff_value = U / 2.0
                else:
                    target_fill = filling_targets[fill_mode]
                    mu_eff_value = U / 2.0

                # iDMRG (always uses grand canonical with fixed mu)
                if include_idmrg:
                    try:
                        meta = {
                            "method": "iDMRG",
                            "U": U,
                            "V": V,
                            "t": t,
                            "L": L,
                            "Nc": None,
                            "int_sep": None,
                            "v_sep": v_sep_ratio,
                            "super_cluster_size": None,
                            "filling_mode": fill_mode,
                        }
                        if compute_localization:
                            energy_dmrg, filling_dmrg, psi, localization = time_call(
                                timing_recorder,
                                meta,
                                run_dmrg_method,
                                U,
                                mu_eff_value,
                                V,
                                v_sep_ratio,
                                t,
                                L,
                                chi,
                                compute_localization=True,
                            )
                            # Store correlation length
                            xi = localization.get('correlation_length') if localization else None
                            if xi is not None and idmrg_corr_length_cache is not None:
                                idmrg_corr_length_cache[u_idx, fill_idx] = xi
                        else:
                            energy_dmrg, filling_dmrg, _ = time_call(
                                timing_recorder,
                                meta,
                                run_dmrg_method,
                                U,
                                mu_eff_value,
                                V,
                                v_sep_ratio,
                                t,
                                L,
                                chi,
                            )
                        dmrg_value = energy_dmrg + mu_eff_value * filling_dmrg
                        idmrg_cache[u_idx, fill_idx] = dmrg_value
                        idmrg_fill_cache[u_idx, fill_idx] = filling_dmrg
                    except Exception as exc:
                        import traceback
                        error_msg = str(exc) or f"{type(exc).__name__}: {repr(exc)}"
                        failed_calculations.append({
                            'method': f'iDMRG ({fill_mode})',
                            'params': {'U': U, 'V': V},
                            'error': error_msg,
                            'traceback': traceback.format_exc(),
                        })

                # Finite DMRG
                if include_finite_dmrg:
                    try:
                        meta = {
                            "method": "DMRG",
                            "U": U,
                            "V": V,
                            "t": t,
                            "L": L,
                            "Nc": None,
                            "int_sep": None,
                            "v_sep": v_sep_ratio,
                            "super_cluster_size": None,
                            "filling_mode": fill_mode,
                        }
                        # In fixed_mu mode or when no target_fill, use grand canonical
                        if use_fixed_mu or target_fill is None:
                            energy_finite, _, filling_finite = time_call(
                                timing_recorder,
                                meta,
                                get_gnd,
                                L,
                                chi,
                                U,
                                t,
                                mu_eff_value,
                                V,
                                v_sep_ratio,
                            )
                            energy_finite_per_site = energy_finite / L
                            dmrg_value = energy_finite_per_site + mu_eff_value * filling_finite
                        elif dmrg_fixed_filling:
                            energy_finite, _, filling_finite = time_call(
                                timing_recorder,
                                meta,
                                get_gnd_fixed_filling,
                                L,
                                chi,
                                target_fill,
                                U,
                                t,
                                V,
                                v_sep_ratio,
                            )
                            energy_finite_per_site = energy_finite / L
                            dmrg_value = energy_finite_per_site
                        else:
                            energy_finite, _, filling_finite = time_call(
                                timing_recorder,
                                meta,
                                get_gnd,
                                L,
                                chi,
                                U,
                                t,
                                mu_eff_value,
                                V,
                                v_sep_ratio,
                            )
                            energy_finite_per_site = energy_finite / L
                            dmrg_value = energy_finite_per_site + mu_eff_value * filling_finite
                        finite_dmrg_cache[u_idx, fill_idx] = dmrg_value
                        finite_dmrg_fill_cache[u_idx, fill_idx] = filling_finite
                    except Exception as exc:
                        import traceback
                        error_msg = str(exc) or f"{type(exc).__name__}: {repr(exc)}"
                        failed_calculations.append({
                            'method': f'Finite DMRG ({fill_mode})',
                            'params': {'U': U, 'V': V},
                            'error': error_msg,
                            'traceback': traceback.format_exc(),
                        })

    # --- PLOTTING LOGIC ---
    # Per-Nc color mapping: maximal separation (1, Nc) is black, others use blue-to-red gradient
    def _blue_to_red_gradient(idx: int, total: int) -> str:
        """Generate color from blue to red gradient based on index."""
        if total <= 1:
            return '#1f77b4'  # Default blue for single item
        # Interpolate from blue to red with good contrast
        t = idx / (total - 1)
        r = int(30 + t * (220 - 30))   # 30 -> 220
        g = int(119 - t * 119)          # 119 -> 0
        b = int(180 - t * 180)          # 180 -> 0
        return f'#{r:02x}{g:02x}{b:02x}'

    def get_color_for_int_sep(int_sep: Tuple[int, int], Nc: int, all_seps_for_nc: List[Tuple[int, int]]) -> str:
        """Get color for an int_sep within a given Nc. Maximal separation (1, Nc) is black."""
        # Maximal separation is (1, Nc) which corresponds to 2π/Nc angular spacing
        if int_sep == (1, Nc):
            return '#000000'  # Black for maximal separation
        # Sort remaining separations by angular value (p/q), exclude the maximal one
        non_maximal = [s for s in all_seps_for_nc if s != (1, Nc)]
        if not non_maximal:
            return '#1f77b4'  # Fallback
        sorted_seps = sorted(non_maximal, key=lambda r: r[0] / r[1], reverse=True)
        if int_sep not in sorted_seps:
            return '#1f77b4'  # Fallback
        idx = sorted_seps.index(int_sep)
        return _blue_to_red_gradient(idx, len(sorted_seps))

    # Build per-Nc color maps
    int_sep_color_by_Nc: Dict[int, Dict[Tuple[int, int], str]] = {}
    for Nc in cluster_sizes:
        seps_for_nc = int_sep_map[Nc]
        int_sep_color_by_Nc[Nc] = {
            sep: get_color_for_int_sep(sep, Nc, seps_for_nc)
            for sep in seps_for_nc
        }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with 2 rows (half/quarter filling) x N_c columns
    n_cols = len(cluster_sizes)
    n_cols_per_page = cols_per_page
    n_pages = int(np.ceil(n_cols / n_cols_per_page))

    figures: List[go.Figure] = []
    html_paths: List[str] = []

    for page_idx in range(n_pages):
        start_col = page_idx * n_cols_per_page
        end_col = min(start_col + n_cols_per_page, n_cols)
        current_cluster_sizes = cluster_sizes[start_col:end_col]
        n_cols_this_page = len(current_cluster_sizes)

        # Subplot titles
        top_titles = [f"Half-filling (n=1): N_c={Nc}" for Nc in current_cluster_sizes]
        bottom_titles = [f"Quarter-filling (n=0.5): N_c={Nc}" for Nc in current_cluster_sizes]

        fig = make_subplots(
            rows=2,
            cols=n_cols_this_page,
            subplot_titles=top_titles + bottom_titles,
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )

        # Build legend name mapping: (row, col) -> 'legend', 'legend2', 'legend3', etc.
        # Plotly requires 'legend' for the first, then 'legend2', 'legend3', ...
        legend_name_map = {}
        legend_idx = 1
        for r in range(1, 3):  # 2 rows
            for c in range(1, n_cols_this_page + 1):
                if legend_idx == 1:
                    legend_name_map[(r, c)] = 'legend'
                else:
                    legend_name_map[(r, c)] = f'legend{legend_idx}'
                legend_idx += 1

        any_trace = False
        for col_idx, Nc in enumerate(current_cluster_sizes):
            col = col_idx + 1

            for row_idx, fill_mode in enumerate(filling_modes):
                row = row_idx + 1

                # Reference data for relative error
                reference_series = finite_dmrg_cache[:, row_idx] if plot_relative_error else None
                if plot_relative_error and reference_series is not None:
                    if not np.any(np.isfinite(reference_series)):
                        warnings.warn(f"No finite DMRG reference for {fill_mode} filling")

                # Add DMRG reference lines (use distinct gold color)
                dmrg_color = '#DAA520'  # Gold
                idmrg_energies = idmrg_cache[:, row_idx]
                if np.any(np.isfinite(idmrg_energies)) and not plot_relative_error:
                    fig.add_trace(
                        go.Scatter(
                            x=u_list,
                            y=idmrg_energies,
                            mode='lines+markers',
                            name="iDMRG",
                            legendgroup=f"iDMRG_{row}_{col}",
                            marker=dict(color=dmrg_color, size=10, symbol='diamond'),
                            line=dict(color=dmrg_color, width=2, dash='dot'),
                            showlegend=True,
                            hovertemplate="U=%{x:.3g}<br>E_iDMRG=%{y:.6f}<extra></extra>",
                            legend=legend_name_map[(row, col)],
                        ),
                        row=row,
                        col=col,
                    )
                    any_trace = True

                finite_dmrg_energies = finite_dmrg_cache[:, row_idx]
                if np.any(np.isfinite(finite_dmrg_energies)) and not plot_relative_error:
                    finite_label = "DMRG"
                    fig.add_trace(
                        go.Scatter(
                            x=u_list,
                            y=finite_dmrg_energies,
                            mode='lines+markers',
                            name=finite_label,
                            legendgroup=f"finite_dmrg_{row}_{col}",
                            marker=dict(color=dmrg_color, size=10, symbol='diamond'),
                            line=dict(color=dmrg_color, width=2, dash='dot'),
                            showlegend=True,
                            hovertemplate="U=%{x:.3g}<br>E_Finite=%{y:.6f}<extra></extra>",
                            legend=legend_name_map[(row, col)],
                        ),
                        row=row,
                        col=col,
                    )
                    any_trace = True

                # Add cluster lines for each interaction separation (specific to this Nc)
                for int_sep in int_sep_map[Nc]:
                    key = (Nc, int_sep, fill_mode)
                    cluster_energies = cluster_results[key]

                    xs, ys, hover_text = [], [], []
                    for u_idx, U in enumerate(u_list):
                        c_en = cluster_energies[u_idx]
                        if not np.isfinite(c_en):
                            continue

                        if plot_relative_error and reference_series is not None:
                            ref = reference_series[u_idx]
                            if not (np.isfinite(ref) and ref != 0):
                                continue
                            c_err = np.abs(c_en - ref) / np.abs(ref)
                            xs.append(U)
                            ys.append(c_err)
                            hover_text.append(
                                f"U={U:.3g}<br>Nc={Nc}<br>int_sep={format_sep_as_pi(int_sep)}<br>rel_err={c_err:.2%}"
                            )
                        else:
                            xs.append(U)
                            ys.append(c_en)
                            hover_text.append(
                                f"U={U:.3g}<br>Nc={Nc}<br>int_sep={format_sep_as_pi(int_sep)}<br>E={c_en:.6f}"
                            )

                    if not xs:
                        continue

                    int_sep_label = format_sep_as_pi(int_sep)
                    trace_color = int_sep_color_by_Nc[Nc][int_sep]
                    fig.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode='lines+markers',
                            name=f"m={int_sep_label}",
                            legendgroup=f"int_sep_{int_sep[0]}_{int_sep[1]}_Nc{Nc}",
                            marker=dict(color=trace_color, size=8),
                            line=dict(color=trace_color, width=2),
                            showlegend=True,  # Show in per-subplot legend
                            hovertemplate="%{text}<extra></extra>",
                            text=hover_text,
                            legend=legend_name_map[(row, col)],  # Per-subplot legend
                        ),
                        row=row,
                        col=col,
                    )
                    any_trace = True

                # Axis labels
                fig.update_xaxes(title_text="U", row=row, col=col)
                fig.update_yaxes(
                    title_text="Relative error" if plot_relative_error else "Energy per site",
                    row=row,
                    col=col,
                    type='linear',
                    tickformat='.2%' if plot_relative_error else '.4f',
                )

        if not any_trace:
            raise RuntimeError("No valid data points available to plot.")

        # Layout and annotations
        v_sep_label = format_sep_as_pi(v_sep_ratio)
        annotation_parts = [
            f"V={V}",
            f"v_sep={v_sep_label}",
            f"t={t}",
            f"L={L}",
            f"chi={chi}",
            f"states={states_retained}",
        ]
        if set_filling is not None:
            annotation_parts.insert(0, f"n_target={set_filling}")
        annotation_text = ", ".join(annotation_parts)
        if n_pages > 1:
            annotation_text += f" | Page {page_idx + 1}/{n_pages}"

        # Calculate subplot domains for legend positioning
        # For a 2-row x n_cols grid with horizontal_spacing=0.08, vertical_spacing=0.12
        h_spacing = 0.08
        v_spacing = 0.12
        subplot_width = (1.0 - h_spacing * (n_cols_this_page - 1)) / n_cols_this_page
        subplot_height = (1.0 - v_spacing) / 2  # 2 rows

        # Build legend configs for each subplot (inset in top-right)
        # Use the same legend_name_map to get correct Plotly legend names
        legend_configs = {}
        for col_idx in range(n_cols_this_page):
            for row_idx in range(2):
                row = row_idx + 1
                col = col_idx + 1
                legend_key = legend_name_map[(row, col)]

                # Calculate subplot position
                x_end = col_idx * (subplot_width + h_spacing) + subplot_width
                y_start = 1.0 - row_idx * (subplot_height + v_spacing) - subplot_height

                # Position legend in bottom-right of subplot
                legend_configs[legend_key] = dict(
                    x=x_end - 0.01,
                    y=y_start + 0.02,
                    xanchor='right',
                    yanchor='bottom',
                    bgcolor='rgba(255, 255, 255, 0.85)',
                    bordercolor='rgba(0, 0, 0, 0.3)',
                    borderwidth=1,
                    font=dict(size=8),
                    itemsizing='constant',
                    tracegroupgap=0,
                    itemwidth=30,  # Thinner legend line samples
                )

        fig.update_layout(
            title=dict(
                text="Relative Error vs U (Half vs Quarter Filling)" if plot_relative_error
                     else "Ground State Energy vs U (Half vs Quarter Filling)",
                x=0.5,
                xanchor='center',
                y=0.98,
                yanchor='top',
            ),
            hovermode='closest',
            height=750,  # Slightly taller to accommodate bottom annotation
            width=400 * n_cols_this_page,
            margin=dict(b=80),  # Extra bottom margin for annotation
            **legend_configs,
        )
        fig.add_annotation(
            text=annotation_text,
            x=0.5,
            xref='paper',
            y=-0.12,
            yref='paper',
            showarrow=False,
            font=dict(size=11, color='gray'),
        )

        if save_html:
            page_suffix = f"_page_{page_idx + 1}" if n_pages > 1 else ""
            html_name = f"{filename_prefix}_L{L}_chi{chi}_{timestamp}{page_suffix}.html"
            html_path = os.path.join(output_dir, html_name)
            fig.write_html(html_path)
            html_paths.append(html_path)
            print(f"Saved figure to {html_path}")

        figures.append(fig)

    # --- SECOND PAGE: FILLING PLOTS ---
    filling_figures: List[go.Figure] = []
    filling_html_paths: List[str] = []

    for page_idx in range(n_pages):
        start_col = page_idx * n_cols_per_page
        end_col = min(start_col + n_cols_per_page, n_cols)
        current_cluster_sizes = cluster_sizes[start_col:end_col]
        n_cols_this_page = len(current_cluster_sizes)

        # Subplot titles for filling
        top_titles_fill = [f"Half-filling (n=1): N_c={Nc}" for Nc in current_cluster_sizes]
        bottom_titles_fill = [f"Quarter-filling (n=0.5): N_c={Nc}" for Nc in current_cluster_sizes]

        fig_fill = make_subplots(
            rows=2,
            cols=n_cols_this_page,
            subplot_titles=top_titles_fill + bottom_titles_fill,
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )

        # Build legend name mapping for filling figure
        legend_name_map_fill = {}
        legend_idx_fill = 1
        for r in range(1, 3):
            for c in range(1, n_cols_this_page + 1):
                if legend_idx_fill == 1:
                    legend_name_map_fill[(r, c)] = 'legend'
                else:
                    legend_name_map_fill[(r, c)] = f'legend{legend_idx_fill}'
                legend_idx_fill += 1

        any_trace_fill = False
        for col_idx, Nc in enumerate(current_cluster_sizes):
            col = col_idx + 1

            for row_idx, fill_mode in enumerate(filling_modes):
                row = row_idx + 1
                target_fill = filling_targets[fill_mode]

                # Add target filling reference line
                fig_fill.add_trace(
                    go.Scatter(
                        x=u_list,
                        y=[target_fill] * len(u_list),
                        mode='lines',
                        name=f"Target n={target_fill}",
                        legendgroup=f"target_{fill_mode}_{row}_{col}",
                        line=dict(color='#aaaaaa', width=1, dash='dash'),
                        showlegend=True,
                        hovertemplate=f"Target filling={target_fill}<extra></extra>",
                        legend=legend_name_map_fill[(row, col)],
                    ),
                    row=row,
                    col=col,
                )
                any_trace_fill = True

                # Add DMRG reference filling lines (use distinct gold color)
                dmrg_color = '#DAA520'  # Gold
                idmrg_fillings = idmrg_fill_cache[:, row_idx]
                if np.any(np.isfinite(idmrg_fillings)):
                    fig_fill.add_trace(
                        go.Scatter(
                            x=u_list,
                            y=idmrg_fillings,
                            mode='lines+markers',
                            name="iDMRG",
                            legendgroup=f"iDMRG_fill_{row}_{col}",
                            marker=dict(color=dmrg_color, size=10, symbol='diamond'),
                            line=dict(color=dmrg_color, width=2, dash='dot'),
                            showlegend=True,
                            hovertemplate="U=%{x:.3g}<br>n_iDMRG=%{y:.4f}<extra></extra>",
                            legend=legend_name_map_fill[(row, col)],
                        ),
                        row=row,
                        col=col,
                    )
                    any_trace_fill = True

                finite_dmrg_fillings = finite_dmrg_fill_cache[:, row_idx]
                if np.any(np.isfinite(finite_dmrg_fillings)):
                    finite_label = "DMRG"
                    fig_fill.add_trace(
                        go.Scatter(
                            x=u_list,
                            y=finite_dmrg_fillings,
                            mode='lines+markers',
                            name=finite_label,
                            legendgroup=f"finite_dmrg_fill_{row}_{col}",
                            marker=dict(color=dmrg_color, size=10, symbol='diamond'),
                            line=dict(color=dmrg_color, width=2, dash='dot'),
                            showlegend=True,
                            hovertemplate="U=%{x:.3g}<br>n_Finite=%{y:.4f}<extra></extra>",
                            legend=legend_name_map_fill[(row, col)],
                        ),
                        row=row,
                        col=col,
                    )
                    any_trace_fill = True

                # Add cluster filling lines for each interaction separation
                for int_sep in int_sep_map[Nc]:
                    key = (Nc, int_sep, fill_mode)
                    cluster_fill_data = cluster_fillings[key]

                    xs, ys, hover_text = [], [], []
                    for u_idx, U in enumerate(u_list):
                        c_fill = cluster_fill_data[u_idx]
                        if not np.isfinite(c_fill):
                            continue
                        xs.append(U)
                        ys.append(c_fill)
                        hover_text.append(
                            f"U={U:.3g}<br>Nc={Nc}<br>int_sep={format_sep_as_pi(int_sep)}<br>n={c_fill:.4f}"
                        )

                    if not xs:
                        continue

                    int_sep_label = format_sep_as_pi(int_sep)
                    trace_color = int_sep_color_by_Nc[Nc][int_sep]
                    fig_fill.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode='lines+markers',
                            name=f"m={int_sep_label}",
                            legendgroup=f"int_sep_{int_sep[0]}_{int_sep[1]}_Nc{Nc}_fill",
                            marker=dict(color=trace_color, size=8),
                            line=dict(color=trace_color, width=2),
                            showlegend=True,
                            hovertemplate="%{text}<extra></extra>",
                            text=hover_text,
                            legend=legend_name_map_fill[(row, col)],
                        ),
                        row=row,
                        col=col,
                    )
                    any_trace_fill = True

                # Axis labels for filling plots
                fig_fill.update_xaxes(title_text="U", row=row, col=col)
                fig_fill.update_yaxes(
                    title_text="Filling per site",
                    row=row,
                    col=col,
                    type='linear',
                    tickformat='.3f',
                    range=[0, 2],
                )

        if not any_trace_fill:
            warnings.warn("No valid filling data points available to plot.")
        else:
            # Calculate subplot domains for legend positioning (same as energy figure)
            h_spacing = 0.08
            v_spacing = 0.12
            subplot_width = (1.0 - h_spacing * (n_cols_this_page - 1)) / n_cols_this_page
            subplot_height = (1.0 - v_spacing) / 2

            legend_configs_fill = {}
            for col_idx_leg in range(n_cols_this_page):
                for row_idx_leg in range(2):
                    row_leg = row_idx_leg + 1
                    col_leg = col_idx_leg + 1
                    legend_key = legend_name_map_fill[(row_leg, col_leg)]

                    x_end = col_idx_leg * (subplot_width + h_spacing) + subplot_width
                    y_start = 1.0 - row_idx_leg * (subplot_height + v_spacing) - subplot_height

                    # Position legend in bottom-right of subplot
                    legend_configs_fill[legend_key] = dict(
                        x=x_end - 0.01,
                        y=y_start + 0.02,
                        xanchor='right',
                        yanchor='bottom',
                        bgcolor='rgba(255, 255, 255, 0.85)',
                        bordercolor='rgba(0, 0, 0, 0.3)',
                        borderwidth=1,
                        font=dict(size=8),
                        itemsizing='constant',
                        tracegroupgap=0,
                        itemwidth=30,  # Thinner legend line samples
                    )

            # Layout for filling figure
            fig_fill.update_layout(
                title=dict(
                    text="Filling per Site vs U (Half vs Quarter Filling)",
                    x=0.5,
                    xanchor='center',
                    y=0.98,
                    yanchor='top',
                ),
                hovermode='closest',
                height=750,  # Slightly taller to accommodate bottom annotation
                width=400 * n_cols_this_page,
                margin=dict(b=80),  # Extra bottom margin for annotation
                **legend_configs_fill,
            )
            fig_fill.add_annotation(
                text=annotation_text,
                x=0.5,
                xref='paper',
                y=-0.12,
                yref='paper',
                showarrow=False,
                font=dict(size=11, color='gray'),
            )

            if save_html:
                page_suffix = f"_page_{page_idx + 1}" if n_pages > 1 else ""
                html_name_fill = f"{filename_prefix}_fillings_L{L}_chi{chi}_{timestamp}{page_suffix}.html"
                html_path_fill = os.path.join(output_dir, html_name_fill)
                fig_fill.write_html(html_path_fill)
                filling_html_paths.append(html_path_fill)
                print(f"Saved filling figure to {html_path_fill}")

            filling_figures.append(fig_fill)

    # --- THIRD PAGE: CORRELATION LENGTH PLOTS (only if compute_localization=True) ---
    corr_length_figures: List[go.Figure] = []
    corr_length_html_paths: List[str] = []

    if compute_localization and idmrg_corr_length_cache is not None:
        for page_idx in range(n_pages):
            start_col = page_idx * n_cols_per_page
            end_col = min(start_col + n_cols_per_page, n_cols)
            current_cluster_sizes = cluster_sizes[start_col:end_col]
            n_cols_this_page = len(current_cluster_sizes)

            # Subplot titles for correlation length
            top_titles_xi = [f"ξ (Half-filling): N_c={Nc}" for Nc in current_cluster_sizes]
            bottom_titles_xi = [f"ξ (Quarter-filling): N_c={Nc}" for Nc in current_cluster_sizes]

            fig_xi = make_subplots(
                rows=2,
                cols=n_cols_this_page,
                subplot_titles=top_titles_xi + bottom_titles_xi,
                horizontal_spacing=0.08,
                vertical_spacing=0.12,
            )

            for col_idx, Nc in enumerate(current_cluster_sizes):
                col = col_idx + 1

                for row_idx, fill_mode in enumerate(filling_modes):
                    row = row_idx + 1

                    # Extract correlation length for this filling mode
                    xi_values = idmrg_corr_length_cache[:, row_idx]
                    valid_mask = ~np.isnan(xi_values)

                    if np.any(valid_mask):
                        fig_xi.add_trace(
                            go.Scatter(
                                x=np.array(u_list)[valid_mask],
                                y=xi_values[valid_mask],
                                mode='lines+markers',
                                name="iDMRG ξ",
                                line=dict(color='#2ca02c', width=2),
                                marker=dict(size=6),
                                showlegend=(col == 1 and row == 1),
                                hovertemplate="U=%{x:.2f}<br>ξ=%{y:.2f}<extra>iDMRG</extra>",
                            ),
                            row=row,
                            col=col,
                        )

                    # Update axes labels
                    fig_xi.update_xaxes(title_text="U" if row == 2 else "", row=row, col=col)
                    fig_xi.update_yaxes(title_text="ξ (correlation length)" if col == 1 else "", row=row, col=col)

            # Layout for correlation length figure
            fig_xi.update_layout(
                title=dict(
                    text="Correlation Length (ξ) vs U",
                    x=0.5,
                    xanchor='center',
                    y=0.98,
                    yanchor='top',
                ),
                hovermode='closest',
                height=750,
                width=400 * n_cols_this_page,
                margin=dict(b=80),
            )
            fig_xi.add_annotation(
                text=annotation_text,
                x=0.5,
                xref='paper',
                y=-0.12,
                yref='paper',
                showarrow=False,
                font=dict(size=11, color='gray'),
            )

            if save_html:
                page_suffix = f"_page_{page_idx + 1}" if n_pages > 1 else ""
                html_name_xi = f"{filename_prefix}_corr_length_L{L}_chi{chi}_{timestamp}{page_suffix}.html"
                html_path_xi = os.path.join(output_dir, html_name_xi)
                fig_xi.write_html(html_path_xi)
                corr_length_html_paths.append(html_path_xi)
                print(f"Saved correlation length figure to {html_path_xi}")

            corr_length_figures.append(fig_xi)

    fig = figures[0] if figures else None
    saved_paths = {}
    if save_html:
        saved_paths['html_pages'] = html_paths
        saved_paths['filling_html_pages'] = filling_html_paths
        saved_paths['corr_length_html_pages'] = corr_length_html_paths
        if html_paths:
            saved_paths['html'] = html_paths[0]
        if filling_html_paths:
            saved_paths['filling_html'] = filling_html_paths[0]
        if corr_length_html_paths:
            saved_paths['corr_length_html'] = corr_length_html_paths[0]

    # Serialize results
    cluster_energy_serialized: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    cluster_fillings_serialized: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for Nc in cluster_sizes:
        cluster_energy_serialized[str(Nc)] = {}
        cluster_fillings_serialized[str(Nc)] = {}
        for int_sep in int_sep_map[Nc]:
            int_sep_key = f"{int_sep[0]}_{int_sep[1]}"
            cluster_energy_serialized[str(Nc)][int_sep_key] = {}
            cluster_fillings_serialized[str(Nc)][int_sep_key] = {}
            for fill_mode in filling_modes:
                key = (Nc, int_sep, fill_mode)
                cluster_energy_serialized[str(Nc)][int_sep_key][fill_mode] = cluster_results[key].tolist()
                cluster_fillings_serialized[str(Nc)][int_sep_key][fill_mode] = cluster_fillings[key].tolist()

    # Serialize int_sep_map for storage
    int_sep_map_serialized = {str(Nc): [list(r) for r in ratios] for Nc, ratios in int_sep_map.items()}

    results_payload = {
        'cluster_sizes': cluster_sizes,
        'int_sep_ratios_by_Nc': int_sep_map_serialized,
        'U_values': u_list,
        'V': V,
        'filling_modes': filling_modes,
        'cluster_energies': cluster_energy_serialized,
        'cluster_fillings': cluster_fillings_serialized,
        'idmrg_energies': idmrg_cache.tolist(),
        'finite_dmrg_energies': finite_dmrg_cache.tolist(),
        'idmrg_fillings': idmrg_fill_cache.tolist(),
        'finite_dmrg_fillings': finite_dmrg_fill_cache.tolist(),
        'idmrg_correlation_lengths': idmrg_corr_length_cache.tolist() if idmrg_corr_length_cache is not None else None,
        'parameters': {
            'v_sep_ratio': v_sep_ratio,
            'V': V,
            't': t,
            'L': L,
            'chi': chi,
            'solver_method': solver_method,
            'states_retained': states_retained,
            'include_idmrg': include_idmrg,
            'include_finite_dmrg': include_finite_dmrg,
            'plot_relative_error': plot_relative_error,
            'set_filling': set_filling,
            'dmrg_fixed_filling': dmrg_fixed_filling,
            'fixed_mu': fixed_mu,
            'compute_localization': compute_localization,
        },
        'artifacts': saved_paths,
        'failures': failed_calculations,
    }

    if include_timing and timing_recorder is not None:
        results_payload['timings'] = timing_recorder.records
        if include_timing_plot and timing_recorder.records:
            fig_timing, timing_artifacts = plot_timings(
                timing_recorder.records,
                output_dir=output_dir,
                filename_prefix=f"{filename_prefix}_timing",
                show_plots=show_plots,
            )
            saved_paths['timing_plot'] = timing_artifacts.get('html')

    if save_data:
        # Include fixed_mu mode in filename if active
        mu_suffix = f"_fixedmu_{fixed_mu}" if fixed_mu else ""
        pickle_name = f"{filename_prefix}_L{L}_chi{chi}{mu_suffix}_{timestamp}.pkl"
        pickle_path = os.path.join(output_dir, pickle_name)
        with open(pickle_path, 'wb') as fh:
            pickle.dump(results_payload, fh)
        saved_paths['pickle'] = pickle_path
        print(f"Saved data to {pickle_path}")

    if failed_calculations:
        print("\n" + "=" * 60)
        print("WARNING: Some calculations failed")
        print("=" * 60)
        for failure in failed_calculations:
            params_desc = ', '.join(f"{k}={v}" for k, v in failure['params'].items())
            print(f"{failure['method']}: {params_desc}")
            error_lines = failure['error'].splitlines()
            print(f"  Error: {error_lines[0] if error_lines else '(no error message)'}")
            if 'traceback' in failure:
                print(f"  Full traceback:\n{failure['traceback']}")

    if show_plots and fig is not None:
        fig.show()
        # Also show filling figures
        for fig_fill in filling_figures:
            fig_fill.show()
        # Also show correlation length figures
        for fig_xi in corr_length_figures:
            fig_xi.show()

    return fig, results_payload


def compare_compressibility_with_int_cluster_sizes(
    int_sep_ratios_by_Nc: Dict[int, Sequence[Tuple[int, int]]],
    U_values: Sequence[float],
    *,
    n_mu_points: int = 30,
    mu_range_factor: float = 2.0,
    mu_min_range: float = 2.0,
    V: float = 0.0,
    v_sep_ratio: Optional[Tuple[int, int]] = None,
    t: float = 1.0,
    L: int = 20,
    chi: int = 32,
    solver_method: str = 'dense_ED',
    states_retained: int = 4,
    output_dir: str = os.path.join(os.path.dirname(__file__), 'large_files', 'plots'),
    show_plots: bool = True,
    save_html: bool = True,
    save_data: bool = True,
    filename_prefix: str = 'compressibility_comparison',
    include_idmrg: bool = True,
    include_finite_dmrg: bool = True,
    plot_relative_error: bool = False,
    results: Optional[Union[Dict, str, os.PathLike]] = None,
    cols_per_page: int = 3,
    rows_per_page: int = 4,
    save_pdf: bool = True,
    axes: Tuple[str, str] = ('U', 'Nc'),
) -> Tuple[go.Figure, Dict]:
    """
    Compressibility plot: filling (n) vs mu_0 at various U values.

    Creates a grid where:
    - X-axis: mu_0 (chemical potential sweep, always)
    - Rows/Columns: U values and cluster sizes N_c (controlled by ``axes``)
    - Lines: different interaction separations for each N_c
    - DMRG reference lines in gold

    For each U, the mu_0 range is generated dynamically as
    [-mu_range_factor * U, mu_range_factor * U], with a minimum half-range of mu_min_range.

    Args:
        int_sep_ratios_by_Nc: Dict mapping each cluster size to compatible int_sep ratios.
        U_values: U values.
        n_mu_points: Number of mu_0 sweep points per U value.
        mu_range_factor: mu_0 range is [-factor*U, factor*U].
        mu_min_range: Minimum half-range for mu_0 when U is small.
        V: Fixed V value.
        v_sep_ratio: V modulation ratio. Defaults to (1,1) when V~0.
        t: Hopping parameter.
        L: System size.
        chi: Bond dimension for DMRG.
        solver_method: Diagonalisation backend.
        states_retained: Number of states retained.
        output_dir: Directory for artifacts.
        show_plots: Whether to display figures.
        save_html: Save interactive HTML figures.
        save_data: Pickle raw results.
        filename_prefix: Prefix for saved files.
        include_idmrg: Include iDMRG reference.
        include_finite_dmrg: Include finite DMRG reference.
        plot_relative_error: Plot relative error vs finite DMRG instead of filling.
        results: Pre-computed results dict or pickle path.
        cols_per_page: Max columns per page.
        rows_per_page: Max rows per page.
        save_pdf: Save publication-quality matplotlib PDF/SVG figures.
        axes: Controls what is on rows vs columns. Default ('U', 'Nc') means
            rows=U values, cols=cluster sizes. Use ('Nc', 'U') for the transpose.

    Returns:
        (figure, results_dict)
    """
    if t is None:
        raise ValueError("Parameter t must be specified.")

    # Handle V≈0 case
    V_ZERO_THRESHOLD = 1e-6
    if v_sep_ratio is None:
        if abs(V) < V_ZERO_THRESHOLD:
            v_sep_ratio = (1, 1)
            print(f"V≈0 detected: using v_sep_ratio=(1,1) to avoid cluster fusion")
        else:
            v_sep_ratio = (1, 2)

    def _coerce_ratio(value, label: str) -> Tuple[int, int]:
        if value is None:
            raise ValueError(f"{label} ratio must be provided.")
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{label} ratio must be a length-2 iterable, got {value!r}.")
        p = int(round(value[0]))
        q = int(round(value[1]))
        if q == 0:
            raise ValueError(f"Denominator for {label} ratio cannot be zero.")
        return (p, q)

    # Validate axes parameter early (doesn't depend on data)
    if not isinstance(axes, (tuple, list)) or len(axes) != 2 or set(axes) != {'U', 'Nc'}:
        raise ValueError(f"axes must be ('U', 'Nc') or ('Nc', 'U'), got {axes!r}")
    row_is_U = (axes[0] == 'U')  # True: rows=U, cols=Nc; False: rows=Nc, cols=U

    # Parse and validate int_sep_ratios_by_Nc (skip if loading from results)
    if results is not None and not int_sep_ratios_by_Nc:
        # Will be reconstructed from pickle data below
        int_sep_map: Dict[int, List[Tuple[int, int]]] = {}
        cluster_sizes: List[int] = []
        u_list: List[float] = []
        mu_arrays: Dict[float, np.ndarray] = {}
        v_sep_ratio = _coerce_ratio(v_sep_ratio, "V separation") if v_sep_ratio is not None else (1, 1)
    else:
        if not int_sep_ratios_by_Nc:
            raise ValueError("Provide at least one cluster size with int_sep_ratios.")

        int_sep_map = {}
        for Nc, ratios in int_sep_ratios_by_Nc.items():
            Nc = int(Nc)
            if not ratios:
                raise ValueError(f"No int_sep_ratios provided for Nc={Nc}.")
            int_sep_map[Nc] = [_coerce_ratio(r, f"int_sep (Nc={Nc})") for r in ratios]

        cluster_sizes = sorted(int_sep_map.keys())
        v_sep_ratio = _coerce_ratio(v_sep_ratio, "V separation")

        U_values_arr = np.asarray(U_values, dtype=float)
        if U_values_arr.ndim != 1 or U_values_arr.size == 0:
            raise ValueError("U_values must be a 1-D array with at least one entry.")
        u_list = [float(u) for u in U_values_arr]

        # Generate per-U mu_0 arrays
        mu_arrays = {}
        for U in u_list:
            half_range = max(mu_range_factor * abs(U), mu_min_range)
            mu_arrays[U] = np.linspace(-half_range, half_range, n_mu_points)

    failed_calculations: List[Dict] = []

    if results is not None:
        # Load from pre-computed results
        if isinstance(results, (str, os.PathLike)):
            results_path = Path(results)
            if not results_path.exists():
                raise ValueError(f"Results file not found: {results_path}")
            with open(results_path, 'rb') as fh:
                results = pickle.load(fh)
        elif not isinstance(results, dict):
            raise ValueError("results must be a dict or path-like object.")

        save_data = False
        print("Using precomputed results; skipping new simulations.")
        params = results.get('parameters', {})

        # Reconstruct mu_arrays from stored data
        stored_mu = results.get('mu_arrays', {})
        mu_arrays = {float(k): np.asarray(v) for k, v in stored_mu.items()}

        # Reconstruct int_sep_map, cluster_sizes, u_list from serialized data if not provided
        serialized_fills = results.get('cluster_fillings', {})
        serialized_energies = results.get('cluster_energies', {})
        if not int_sep_map:
            # Rebuild from serialized keys: cluster_fillings[str(Nc)][p_q][str(U)]
            for nc_key, sep_dict in serialized_fills.items():
                nc = int(nc_key)
                seps = []
                for sep_key in sep_dict:
                    p, q = sep_key.split('_')
                    seps.append((int(p), int(q)))
                int_sep_map[nc] = seps
            cluster_sizes = sorted(int_sep_map.keys())
        if not u_list:
            u_list = sorted(mu_arrays.keys())

        # Load cluster fillings and energies
        cluster_fillings: Dict[Tuple[int, Tuple[int, int], float], np.ndarray] = {}
        cluster_energies: Dict[Tuple[int, Tuple[int, int], float], np.ndarray] = {}

        for Nc in cluster_sizes:
            for int_sep in int_sep_map[Nc]:
                int_sep_key = f"{int_sep[0]}_{int_sep[1]}"
                for U in u_list:
                    u_key = str(U)
                    key = (Nc, int_sep, U)
                    n_mu = len(mu_arrays[U])

                    fill_by_nc = serialized_fills.get(str(Nc), {})
                    fill_by_sep = fill_by_nc.get(int_sep_key, {})
                    fill_series = fill_by_sep.get(u_key)
                    cluster_fillings[key] = np.asarray(fill_series, dtype=float) if fill_series is not None else np.full(n_mu, np.nan)

                    en_by_nc = serialized_energies.get(str(Nc), {})
                    en_by_sep = en_by_nc.get(int_sep_key, {})
                    en_series = en_by_sep.get(u_key)
                    cluster_energies[key] = np.asarray(en_series, dtype=float) if en_series is not None else np.full(n_mu, np.nan)

        # Load DMRG caches
        idmrg_fill_cache: Dict[float, np.ndarray] = {}
        finite_dmrg_fill_cache: Dict[float, np.ndarray] = {}
        stored_idmrg = results.get('idmrg_fillings', {})
        stored_finite = results.get('finite_dmrg_fillings', {})
        for U in u_list:
            u_key = str(U)
            n_mu = len(mu_arrays[U])
            idmrg_fill_cache[U] = np.asarray(stored_idmrg.get(u_key, []), dtype=float) if stored_idmrg.get(u_key) is not None else np.full(n_mu, np.nan)
            finite_dmrg_fill_cache[U] = np.asarray(stored_finite.get(u_key, []), dtype=float) if stored_finite.get(u_key) is not None else np.full(n_mu, np.nan)

        # Override metadata
        v_sep_ratio = _coerce_ratio(params.get('v_sep_ratio', v_sep_ratio), "V separation")
        t = params.get('t', t)
        L = params.get('L', L)
        chi = params.get('chi', chi)
        V = params.get('V', V)
        states_retained = params.get('states_retained', states_retained)
        include_idmrg = params.get('include_idmrg', include_idmrg)
        include_finite_dmrg = params.get('include_finite_dmrg', include_finite_dmrg)
    else:
        # Compute fresh results
        cluster_fillings = {
            (Nc, int_sep, U): np.full(n_mu_points, np.nan, dtype=float)
            for Nc in cluster_sizes
            for int_sep in int_sep_map[Nc]
            for U in u_list
        }
        cluster_energies = {
            (Nc, int_sep, U): np.full(n_mu_points, np.nan, dtype=float)
            for Nc in cluster_sizes
            for int_sep in int_sep_map[Nc]
            for U in u_list
        }
        idmrg_fill_cache = {U: np.full(n_mu_points, np.nan, dtype=float) for U in u_list}
        finite_dmrg_fill_cache = {U: np.full(n_mu_points, np.nan, dtype=float) for U in u_list}

        print("=" * 60)
        print("Running compressibility calculations")
        print(f"Cluster sizes: {cluster_sizes}")
        for Nc in cluster_sizes:
            print(f"  Nc={Nc}: {[format_sep_as_pi(r) for r in int_sep_map[Nc]]}")
        print(f"V = {V}, v_sep = {format_sep_as_pi(v_sep_ratio)}")
        print(f"U values: {u_list}")
        print(f"mu_0 points per U: {n_mu_points}")
        print("=" * 60)

        for Nc in tqdm(cluster_sizes, desc="Cluster sizes", ncols=80):
            for int_sep in int_sep_map[Nc]:
                for U in u_list:
                    mu_arr = mu_arrays[U]
                    key = (Nc, int_sep, U)

                    # --- Batched mu_0 sweep (GPU-accelerated) ---
                    # Diagonalise ONCE at mu_0=0.  In each fixed-(Nup,Ndn)
                    # sector the mu_0 term is just -mu_0*(Nup+Ndn)*I, so
                    # eigenvalues shift analytically and eigenvectors are
                    # unchanged.  The Boltzmann weighting over all mu_0
                    # values is then a single vectorised (optionally GPU)
                    # operation.
                    # NOTE: this optimisation applies to the cluster ED
                    # path only — DMRG must still sweep mu_0 pointwise
                    # because it finds one ground state whose identity
                    # changes with mu_0.
                    try:
                        physical_params_ref = PhysicalParams(U=U, mu_0=0.0, V=V, t=t)
                        run_config_ref = ClusterModelConfig(
                            L=L,
                            int_cluster_size=Nc,
                            cluster_separation_ratio=int_sep,
                            V_separation_ratio=v_sep_ratio,
                            ham_lib='quspin',
                            physical_params=physical_params_ref,
                            model_bc='periodic',
                            int_cluster_bc='periodic',
                            super_cluster_bc='periodic',
                            solver_method=solver_method,
                            states_retained=states_retained,
                        )

                        # Diagonalise all number sectors once (mu_0 = 0)
                        _kpts, energy_ref, number_spec, _spin_spec = get_general_spectra(run_config_ref)
                        # energy_ref:   (n_sc, n_states)
                        # number_spec:  (n_sc, n_states, sc_size)
                        particle_numbers = number_spec.sum(axis=-1)  # (n_sc, n_states)

                        # Vectorised Boltzmann weighting for every mu_0 at once (GPU if available)
                        sys_energies, sys_fillings = batched_mu_expectations(
                            energy_ref, particle_numbers, mu_arr, temperature=1e-2,
                        )

                        cluster_fillings[key] = sys_fillings / L
                        cluster_energies[key] = (sys_energies + mu_arr * sys_fillings) / L
                    except Exception as exc:
                        import traceback
                        error_msg = str(exc) or f"{type(exc).__name__}: {repr(exc)}"
                        failed_calculations.append({
                            'method': f'cluster Nc={Nc}, int_sep={format_sep_as_pi(int_sep)}',
                            'params': {'U': U, 'V': V},
                            'error': error_msg,
                            'traceback': traceback.format_exc(),
                        })

        # DMRG reference sweeps
        print("\n" + "=" * 60)
        print("Computing DMRG references")
        print("=" * 60)

        for U in u_list:
            mu_arr = mu_arrays[U]
            for mu_idx, mu_0 in enumerate(tqdm(mu_arr, desc=f"DMRG U={U}", ncols=80)):
                if include_idmrg:
                    try:
                        energy_dmrg, filling_dmrg, _ = run_dmrg_method(
                            U, mu_0, V, v_sep_ratio, t, L, chi,
                        )
                        idmrg_fill_cache[U][mu_idx] = filling_dmrg
                    except Exception as exc:
                        import traceback
                        failed_calculations.append({
                            'method': 'iDMRG',
                            'params': {'U': U, 'V': V, 'mu_0': mu_0},
                            'error': str(exc),
                            'traceback': traceback.format_exc(),
                        })

                if include_finite_dmrg:
                    try:
                        energy_finite, _, filling_finite = get_gnd(
                            L, chi, U, t, mu_0, V, v_sep_ratio,
                        )
                        finite_dmrg_fill_cache[U][mu_idx] = filling_finite
                    except Exception as exc:
                        import traceback
                        failed_calculations.append({
                            'method': 'Finite DMRG',
                            'params': {'U': U, 'V': V, 'mu_0': mu_0},
                            'error': str(exc),
                            'traceback': traceback.format_exc(),
                        })

    # --- COLOR MAPPING ---
    def _blue_to_red_gradient(idx: int, total: int) -> str:
        if total <= 1:
            return '#1f77b4'
        frac = idx / (total - 1)
        r = int(30 + frac * (220 - 30))
        g = int(119 - frac * 119)
        b = int(180 - frac * 180)
        return f'#{r:02x}{g:02x}{b:02x}'

    def get_color_for_int_sep(int_sep: Tuple[int, int], Nc: int, all_seps: List[Tuple[int, int]]) -> str:
        if int_sep == (1, Nc):
            return '#000000'
        non_maximal = [s for s in all_seps if s != (1, Nc)]
        if not non_maximal:
            return '#1f77b4'
        sorted_seps = sorted(non_maximal, key=lambda r: r[0] / r[1], reverse=True)
        if int_sep not in sorted_seps:
            return '#1f77b4'
        idx = sorted_seps.index(int_sep)
        return _blue_to_red_gradient(idx, len(sorted_seps))

    int_sep_color_by_Nc: Dict[int, Dict[Tuple[int, int], str]] = {}
    for Nc in cluster_sizes:
        seps = int_sep_map[Nc]
        int_sep_color_by_Nc[Nc] = {sep: get_color_for_int_sep(sep, Nc, seps) for sep in seps}

    # --- PLOTTING ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)

    # Set up row/col mapping based on axes parameter
    if row_is_U:
        row_values: list = u_list          # rows index U
        col_values: list = cluster_sizes   # cols index Nc
    else:
        row_values = cluster_sizes         # rows index Nc
        col_values = u_list                # cols index U

    n_rows_total = len(row_values)
    n_cols_total = len(col_values)
    n_cols_pp = min(cols_per_page, n_cols_total)
    n_rows_pp = min(rows_per_page, n_rows_total)
    n_col_pages = int(np.ceil(n_cols_total / n_cols_pp))
    n_row_pages = int(np.ceil(n_rows_total / n_rows_pp))
    n_pages = n_col_pages * n_row_pages

    def _get_U_Nc(ri: int, ci: int) -> Tuple[float, int]:
        """Return (U, Nc) for a given row/col index into the current page's values."""
        if row_is_U:
            return current_row_values[ri], current_col_values[ci]
        else:
            return current_col_values[ci], current_row_values[ri]

    def _row_label(val) -> str:
        if row_is_U:
            return f"U={val}"
        return f"N_c={val}"

    def _col_label(val) -> str:
        if row_is_U:
            return f"N_c={val}"
        return f"U={val}"

    figures: List[go.Figure] = []
    html_paths: List[str] = []

    for page_idx in range(n_pages):
        row_page = page_idx // n_col_pages
        col_page = page_idx % n_col_pages

        row_start = row_page * n_rows_pp
        row_end = min(row_start + n_rows_pp, n_rows_total)
        col_start = col_page * n_cols_pp
        col_end = min(col_start + n_cols_pp, n_cols_total)

        current_row_values = row_values[row_start:row_end]
        current_col_values = col_values[col_start:col_end]
        n_rows_this = len(current_row_values)
        n_cols_this = len(current_col_values)

        subplot_titles = []
        for rv in current_row_values:
            for cv in current_col_values:
                subplot_titles.append(f"{_row_label(rv)}, {_col_label(cv)}")

        fig = make_subplots(
            rows=n_rows_this,
            cols=n_cols_this,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.08,
            vertical_spacing=0.12 / max(n_rows_this - 1, 1) if n_rows_this > 1 else 0.12,
        )

        # Per-subplot legend mapping
        legend_name_map = {}
        legend_idx = 1
        for r in range(1, n_rows_this + 1):
            for c in range(1, n_cols_this + 1):
                legend_name_map[(r, c)] = 'legend' if legend_idx == 1 else f'legend{legend_idx}'
                legend_idx += 1

        any_trace = False
        for row_idx in range(n_rows_this):
            row = row_idx + 1

            for col_idx in range(n_cols_this):
                col = col_idx + 1
                U, Nc = _get_U_Nc(row_idx, col_idx)
                mu_arr = mu_arrays[U]

                # DMRG reference lines
                dmrg_color = '#DAA520'
                if include_idmrg and np.any(np.isfinite(idmrg_fill_cache[U])) and not plot_relative_error:
                    fig.add_trace(
                        go.Scatter(
                            x=mu_arr.tolist(),
                            y=idmrg_fill_cache[U].tolist(),
                            mode='lines+markers',
                            name="iDMRG",
                            legendgroup=f"iDMRG_{row}_{col}",
                            marker=dict(color=dmrg_color, size=10, symbol='diamond'),
                            line=dict(color=dmrg_color, width=2, dash='dot'),
                            showlegend=True,
                            hovertemplate="μ=%{x:.3g}<br>n_iDMRG=%{y:.4f}<extra></extra>",
                            legend=legend_name_map[(row, col)],
                        ),
                        row=row, col=col,
                    )
                    any_trace = True

                if include_finite_dmrg and np.any(np.isfinite(finite_dmrg_fill_cache[U])) and not plot_relative_error:
                    fig.add_trace(
                        go.Scatter(
                            x=mu_arr.tolist(),
                            y=finite_dmrg_fill_cache[U].tolist(),
                            mode='lines+markers',
                            name="DMRG",
                            legendgroup=f"finite_dmrg_{row}_{col}",
                            marker=dict(color=dmrg_color, size=10, symbol='diamond'),
                            line=dict(color=dmrg_color, width=2, dash='dash'),
                            showlegend=True,
                            hovertemplate="μ=%{x:.3g}<br>n_DMRG=%{y:.4f}<extra></extra>",
                            legend=legend_name_map[(row, col)],
                        ),
                        row=row, col=col,
                    )
                    any_trace = True

                # Cluster lines for each int_sep
                reference_series = finite_dmrg_fill_cache[U] if plot_relative_error else None

                for int_sep in int_sep_map[Nc]:
                    key = (Nc, int_sep, U)
                    fills = cluster_fillings[key]

                    xs, ys, hover_text = [], [], []
                    for mu_idx, mu_0 in enumerate(mu_arr):
                        f_val = fills[mu_idx]
                        if not np.isfinite(f_val):
                            continue

                        if plot_relative_error and reference_series is not None:
                            ref = reference_series[mu_idx]
                            if not (np.isfinite(ref) and ref != 0):
                                continue
                            rel_err = np.abs(f_val - ref) / np.abs(ref)
                            xs.append(mu_0)
                            ys.append(rel_err)
                            hover_text.append(
                                f"μ={mu_0:.3g}<br>Nc={Nc}<br>m={format_sep_as_pi(int_sep)}<br>rel_err={rel_err:.2%}"
                            )
                        else:
                            xs.append(mu_0)
                            ys.append(f_val)
                            hover_text.append(
                                f"μ={mu_0:.3g}<br>Nc={Nc}<br>m={format_sep_as_pi(int_sep)}<br>n={f_val:.4f}"
                            )

                    if not xs:
                        continue

                    trace_color = int_sep_color_by_Nc[Nc][int_sep]
                    fig.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode='lines+markers',
                            name=f"m={format_sep_as_pi(int_sep)}",
                            legendgroup=f"int_sep_{int_sep[0]}_{int_sep[1]}_Nc{Nc}",
                            marker=dict(color=trace_color, size=8),
                            line=dict(color=trace_color, width=2),
                            showlegend=True,
                            hovertemplate="%{text}<extra></extra>",
                            text=hover_text,
                            legend=legend_name_map[(row, col)],
                        ),
                        row=row, col=col,
                    )
                    any_trace = True

                # Axis labels
                fig.update_xaxes(title_text="μ₀", row=row, col=col)
                fig.update_yaxes(
                    title_text="Relative error" if plot_relative_error else "Filling per site (n)",
                    row=row, col=col,
                    type='linear',
                    tickformat='.2%' if plot_relative_error else '.3f',
                )

        if not any_trace:
            warnings.warn("No valid data points to plot on this page.")
            continue

        # Layout
        v_sep_label = format_sep_as_pi(v_sep_ratio)
        annotation_text = f"V={V}, v_sep={v_sep_label}, t={t}, L={L}, chi={chi}, states={states_retained}"
        if n_pages > 1:
            annotation_text += f" | Page {page_idx + 1}/{n_pages}"

        # Per-subplot legend positioning
        h_spacing = 0.08
        v_spacing_frac = 0.12
        subplot_width = (1.0 - h_spacing * (n_cols_this - 1)) / n_cols_this
        subplot_height = (1.0 - v_spacing_frac * (n_rows_this - 1)) / n_rows_this if n_rows_this > 1 else 1.0

        legend_configs = {}
        for ci in range(n_cols_this):
            for ri in range(n_rows_this):
                lk = legend_name_map[(ri + 1, ci + 1)]
                x_end = ci * (subplot_width + h_spacing) + subplot_width
                y_start = 1.0 - ri * (subplot_height + v_spacing_frac) - subplot_height if n_rows_this > 1 else 0.0
                legend_configs[lk] = dict(
                    x=x_end - 0.01,
                    y=y_start + 0.02,
                    xanchor='right',
                    yanchor='bottom',
                    bgcolor='rgba(255, 255, 255, 0.85)',
                    bordercolor='rgba(0, 0, 0, 0.3)',
                    borderwidth=1,
                    font=dict(size=8),
                    itemsizing='constant',
                    tracegroupgap=0,
                    itemwidth=30,
                )

        fig.update_layout(
            title=dict(
                text="Compressibility: Filling vs μ₀" if not plot_relative_error
                     else "Compressibility: Relative Error vs μ₀",
                x=0.5, xanchor='center', y=0.98, yanchor='top',
            ),
            hovermode='closest',
            height=350 * n_rows_this,
            width=400 * n_cols_this,
            margin=dict(b=80),
            **legend_configs,
        )
        fig.add_annotation(
            text=annotation_text,
            x=0.5, xref='paper', y=-0.06, yref='paper',
            showarrow=False, font=dict(size=11, color='gray'),
        )

        if save_html:
            page_suffix = f"_page_{page_idx + 1}" if n_pages > 1 else ""
            html_name = f"{filename_prefix}_L{L}_chi{chi}_{timestamp}{page_suffix}.html"
            html_path = os.path.join(output_dir, html_name)
            fig.write_html(html_path)
            html_paths.append(html_path)
            print(f"Saved figure to {html_path}")

        figures.append(fig)

    # --- MATPLOTLIB FIGURES ---
    mpl_paths: List[str] = []
    if save_pdf:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 11,
            'legend.fontsize': 7,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'lines.linewidth': 1.5,
            'lines.markersize': 4,
            'axes.linewidth': 0.8,
            'grid.linewidth': 0.4,
            'grid.alpha': 0.3,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
        })
        # Use tex if available, fall back gracefully
        try:
            plt.rcParams.update({
                'text.usetex': True,
                'text.latex.preamble': r'\usepackage{amsmath}',
            })
            # Quick test to see if LaTeX works
            _test_fig, _test_ax = plt.subplots(1, 1, figsize=(1, 1))
            _test_ax.set_xlabel(r"$\mu$")
            _test_fig.savefig(os.path.join(output_dir, "_latex_test.pdf"))
            plt.close(_test_fig)
            os.remove(os.path.join(output_dir, "_latex_test.pdf"))
            use_tex = True
        except Exception:
            plt.rcParams['text.usetex'] = False
            use_tex = False

        def _pi_label(sep_str: str) -> str:
            """Convert format_sep_as_pi output to LaTeX or plain text."""
            if use_tex:
                return '$' + sep_str.replace('π', r'\pi') + '$'
            return sep_str

        mpl_n_rows = len(row_values)
        mpl_n_cols = len(col_values)
        v_sep_label = format_sep_as_pi(v_sep_ratio)
        v_sep_latex = v_sep_label.replace('π', r'\pi') if use_tex else v_sep_label

        # Generate both filling and relative-error figures
        plot_modes = [False]  # Always produce filling plot
        if include_finite_dmrg:
            plot_modes.append(True)  # Also produce relative error plot if DMRG reference exists

        for is_rel_error in plot_modes:
            mode_tag = 'rel_error' if is_rel_error else 'filling'

            fig_width = 3.2 * mpl_n_cols + 0.6
            fig_height = 2.8 * mpl_n_rows + 0.8
            mpl_fig, axs = plt.subplots(
                mpl_n_rows, mpl_n_cols,
                figsize=(fig_width, fig_height),
                squeeze=False,
                sharex=False,
            )

            for ri in range(mpl_n_rows):
                for ci in range(mpl_n_cols):
                    if row_is_U:
                        U, Nc = u_list[ri], cluster_sizes[ci]
                    else:
                        U, Nc = u_list[ci], cluster_sizes[ri]
                    mu_arr = mu_arrays[U]
                    ax = axs[ri, ci]

                    # DMRG reference lines (only on filling plot)
                    dmrg_color = '#B8860B'  # Dark goldenrod
                    if not is_rel_error:
                        if include_idmrg and np.any(np.isfinite(idmrg_fill_cache[U])):
                            ax.plot(
                                mu_arr, idmrg_fill_cache[U],
                                color=dmrg_color, ls=':', lw=2.0,
                                marker='D', ms=5, markerfacecolor='none', markeredgewidth=0.8,
                                label='iDMRG', zorder=10,
                            )
                        if include_finite_dmrg and np.any(np.isfinite(finite_dmrg_fill_cache[U])):
                            ax.plot(
                                mu_arr, finite_dmrg_fill_cache[U],
                                color=dmrg_color, ls='--', lw=2.0,
                                marker='D', ms=5, markerfacecolor='none', markeredgewidth=0.8,
                                label='DMRG', zorder=10,
                            )

                    # Cluster lines
                    reference_series = finite_dmrg_fill_cache[U] if is_rel_error else None
                    REL_ERR_REF_THRESHOLD = 0.02  # Skip points where DMRG filling < this
                    for int_sep in int_sep_map[Nc]:
                        key = (Nc, int_sep, U)
                        fills = cluster_fillings[key]
                        trace_color = int_sep_color_by_Nc[Nc][int_sep]
                        sep_label = format_sep_as_pi(int_sep)
                        is_maximal = (int_sep == (1, Nc))

                        xs, ys = [], []
                        for mu_idx, mu_0 in enumerate(mu_arr):
                            f_val = fills[mu_idx]
                            if not np.isfinite(f_val):
                                continue
                            if is_rel_error and reference_series is not None:
                                ref = reference_series[mu_idx]
                                if not (np.isfinite(ref) and abs(ref) > REL_ERR_REF_THRESHOLD):
                                    continue
                                xs.append(mu_0)
                                ys.append(np.abs(f_val - ref) / np.abs(ref))
                            else:
                                xs.append(mu_0)
                                ys.append(f_val)

                        if not xs:
                            continue

                        ax.plot(
                            xs, ys,
                            color=trace_color,
                            ls='-',
                            lw=2.0 if is_maximal else 1.3,
                            marker='o' if is_maximal else 's',
                            ms=4 if is_maximal else 3,
                            label=f'm={_pi_label(sep_label)}',
                            zorder=5 if is_maximal else 3,
                        )

                    # Formatting
                    ax.set_xlabel(r'$\mu_0$' if use_tex else 'mu_0')
                    if ci == 0:
                        row_label = (f'$U={U:g}$' if use_tex else f'U={U:g}') if row_is_U else (f'$N_c={Nc}$' if use_tex else f'Nc={Nc}')
                        if is_rel_error:
                            ylabel = r'Rel.\ error in $n$' if use_tex else 'Rel. error in n'
                        else:
                            ylabel = r'$n$ (filling/site)' if use_tex else 'n (filling/site)'
                        ax.set_ylabel(f'{row_label}\n{ylabel}')
                    if ri == 0:
                        col_title = (f'$N_c = {Nc}$' if use_tex else f'Nc = {Nc}') if row_is_U else (f'$U = {U:g}$' if use_tex else f'U = {U:g}')
                        ax.set_title(col_title)

                    ax.grid(True, ls='--', alpha=0.3)
                    if not is_rel_error:
                        ax.set_ylim(-0.05, 2.05)
                    # Set x-limits to the mu range for this U row
                    margin = 0.05 * (mu_arr[-1] - mu_arr[0])
                    ax.set_xlim(mu_arr[0] - margin, mu_arr[-1] + margin)
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

                    # Legend on every subplot
                    ax.legend(
                        loc='upper right',
                        framealpha=0.85,
                        edgecolor='0.7', handlelength=1.5,
                        borderpad=0.3, labelspacing=0.25,
                        fontsize=6,
                    )

            # Suptitle
            if is_rel_error:
                suptitle = (
                    f'Relative Error in Filling: $V={V}$, $v_{{\\mathrm{{sep}}}}={v_sep_latex}$, $t={t}$, $L={L}$, $\\chi={chi}$'
                    if use_tex else
                    f'Relative Error in Filling: V={V}, v_sep={v_sep_label}, t={t}, L={L}, chi={chi}'
                )
            else:
                suptitle = (
                    f'Compressibility: $V={V}$, $v_{{\\mathrm{{sep}}}}={v_sep_latex}$, $t={t}$, $L={L}$, $\\chi={chi}$'
                    if use_tex else
                    f'Compressibility: V={V}, v_sep={v_sep_label}, t={t}, L={L}, chi={chi}'
                )
            mpl_fig.suptitle(suptitle, fontsize=13, y=1.01)
            mpl_fig.tight_layout(rect=[0, 0, 0.96, 1.0])

            # Save PDF and SVG
            for ext in ('pdf', 'svg'):
                mpl_name = f"{filename_prefix}_{mode_tag}_L{L}_chi{chi}_{timestamp}.{ext}"
                mpl_path = os.path.join(output_dir, mpl_name)
                mpl_fig.savefig(mpl_path, format=ext)
                mpl_paths.append(mpl_path)
                print(f"Saved matplotlib figure to {mpl_path}")

            if show_plots:
                plt.show()
            else:
                plt.close(mpl_fig)

    # --- SERIALIZATION ---
    fig = figures[0] if figures else None
    saved_paths: Dict[str, Any] = {}
    if save_html:
        saved_paths['html_pages'] = html_paths
        if html_paths:
            saved_paths['html'] = html_paths[0]
    if mpl_paths:
        saved_paths['mpl_figures'] = mpl_paths

    cluster_fillings_serialized: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    cluster_energies_serialized: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for Nc in cluster_sizes:
        cluster_fillings_serialized[str(Nc)] = {}
        cluster_energies_serialized[str(Nc)] = {}
        for int_sep in int_sep_map[Nc]:
            int_sep_key = f"{int_sep[0]}_{int_sep[1]}"
            cluster_fillings_serialized[str(Nc)][int_sep_key] = {}
            cluster_energies_serialized[str(Nc)][int_sep_key] = {}
            for U in u_list:
                u_key = str(U)
                key = (Nc, int_sep, U)
                cluster_fillings_serialized[str(Nc)][int_sep_key][u_key] = cluster_fillings[key].tolist()
                cluster_energies_serialized[str(Nc)][int_sep_key][u_key] = cluster_energies[key].tolist()

    mu_arrays_serialized = {str(U): mu_arrays[U].tolist() for U in u_list}
    idmrg_serialized = {str(U): idmrg_fill_cache[U].tolist() for U in u_list}
    finite_dmrg_serialized = {str(U): finite_dmrg_fill_cache[U].tolist() for U in u_list}

    int_sep_map_serialized = {str(Nc): [list(r) for r in ratios] for Nc, ratios in int_sep_map.items()}

    results_payload = {
        'cluster_sizes': cluster_sizes,
        'int_sep_ratios_by_Nc': int_sep_map_serialized,
        'U_values': u_list,
        'V': V,
        'mu_arrays': mu_arrays_serialized,
        'cluster_fillings': cluster_fillings_serialized,
        'cluster_energies': cluster_energies_serialized,
        'idmrg_fillings': idmrg_serialized,
        'finite_dmrg_fillings': finite_dmrg_serialized,
        'parameters': {
            'v_sep_ratio': v_sep_ratio,
            'V': V,
            't': t,
            'L': L,
            'chi': chi,
            'solver_method': solver_method,
            'states_retained': states_retained,
            'include_idmrg': include_idmrg,
            'include_finite_dmrg': include_finite_dmrg,
            'n_mu_points': n_mu_points,
            'mu_range_factor': mu_range_factor,
            'mu_min_range': mu_min_range,
        },
        'artifacts': saved_paths,
        'failures': failed_calculations,
    }

    if save_data:
        pickle_name = f"{filename_prefix}_L{L}_chi{chi}_{timestamp}.pkl"
        pickle_path = os.path.join(output_dir, pickle_name)
        with open(pickle_path, 'wb') as fh:
            pickle.dump(results_payload, fh)
        saved_paths['pickle'] = pickle_path
        print(f"Saved data to {pickle_path}")

    if failed_calculations:
        print("\n" + "=" * 60)
        print("WARNING: Some calculations failed")
        print("=" * 60)
        for failure in failed_calculations:
            params_desc = ', '.join(f"{k}={v}" for k, v in failure['params'].items())
            print(f"{failure['method']}: {params_desc}")
            error_lines = failure['error'].splitlines()
            print(f"  Error: {error_lines[0] if error_lines else '(no error message)'}")

    if show_plots and fig is not None:
        for f in figures:
            f.show()

    return fig, results_payload


def compare_compressibility_cluster_sizes(
    v_sep_ratio: Tuple[int, int],
    int_sep_ratios: Union[Tuple[int, int], Dict[int, Tuple[int, int]]],
    cluster_sizes: Sequence[int],
    U_values: Sequence[float],
    V_values: Sequence[float],
    *,
    n_mu_points: int = 30,
    mu_range_factor: float = 2.0,
    mu_min_range: float = 2.0,
    t: float = 1.0,
    L: int = 20,
    chi: int = 32,
    solver_method: str = 'dense_ED',
    states_retained: int = 4,
    output_dir: str = os.path.join(os.path.dirname(__file__), 'large_files', 'plots'),
    show_plots: bool = True,
    save_html: bool = True,
    save_data: bool = True,
    save_pdf: bool = True,
    filename_prefix: str = 'compressibility_cluster_sizes',
    include_idmrg: bool = True,
    include_finite_dmrg: bool = True,
    results: Optional[Union[Dict, str, os.PathLike]] = None,
    axes: Tuple[str, str] = ('U', 'V'),
    cols_per_page: int = 3,
    rows_per_page: int = 4,
) -> Tuple[go.Figure, Dict]:
    """
    Compressibility plot comparing cluster sizes for non-zero V.

    Sweeps chemical potential mu_0 on the x-axis and plots filling per site (n)
    on the y-axis. Each subplot corresponds to a fixed (U, V) pair, with
    different cluster sizes (Nc) shown as separate lines.

    Args:
        v_sep_ratio: Ratio controlling the AA modulation for V.
        int_sep_ratios: Either a single (p, q) tuple applied to every cluster
            size or a dict mapping each Nc to its specific int_sep ratio.
        cluster_sizes: Cluster sizes; each becomes a separate line.
        U_values: U values (Hubbard interaction strengths).
        V_values: V values (AA modulation strengths).
        n_mu_points: Number of mu_0 sweep points per (U, V) pair.
        mu_range_factor: mu_0 range = [-factor*(|U|+|V|), factor*(|U|+|V|)].
        mu_min_range: Minimum half-range for mu_0 when U+V is small.
        axes: Controls grid layout. ('U', 'V') -> rows=U, cols=V.
            ('V', 'U') -> rows=V, cols=U.
        Other args: Same as compare_filling_cluster_sizes.

    Returns:
        (figure, results_dict)
    """
    if t is None:
        raise ValueError("Parameter t must be specified.")

    def _coerce_ratio(value, label: str) -> Tuple[int, int]:
        if value is None:
            raise ValueError(f"{label} ratio must be provided.")
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{label} ratio must be a length-2 iterable, got {value!r}.")
        p = int(round(value[0]))
        q = int(round(value[1]))
        if q == 0:
            raise ValueError(f"Denominator for {label} ratio cannot be zero.")
        return (p, q)

    # Validate axes parameter
    if not isinstance(axes, (tuple, list)) or len(axes) != 2 or set(axes) != {'U', 'V'}:
        raise ValueError(f"axes must be ('U', 'V') or ('V', 'U'), got {axes!r}")
    row_is_U = (axes[0] == 'U')

    # Parse inputs (skip validation when loading from pickle with empty inputs)
    if results is not None and not cluster_sizes:
        cluster_sizes_list: List[int] = []
        u_list: List[float] = []
        v_list: List[float] = []
        ratio_map: Dict[int, Tuple[int, int]] = {}
        mu_arrays: Dict[Tuple[float, float], np.ndarray] = {}
        v_sep_ratio = _coerce_ratio(v_sep_ratio, "V separation") if v_sep_ratio is not None else (1, 1)
    else:
        cluster_sizes_list = sorted({int(size) for size in cluster_sizes})
        if not cluster_sizes_list:
            raise ValueError("Provide at least one cluster size (N_c).")

        U_arr = np.asarray(U_values, dtype=float)
        V_arr = np.asarray(V_values, dtype=float)
        if U_arr.ndim != 1 or U_arr.size == 0:
            raise ValueError("U_values must be a 1-D array with at least one entry.")
        if V_arr.ndim != 1 or V_arr.size == 0:
            raise ValueError("V_values must be a 1-D array with at least one entry.")
        u_list = [float(u) for u in U_arr]
        v_list = [float(v) for v in V_arr]

        v_sep_ratio = _coerce_ratio(v_sep_ratio, "V separation")

        # Build ratio_map
        if isinstance(int_sep_ratios, dict):
            ratio_map = {}
            fallback_ratio: Optional[Tuple[int, int]] = None
            for v in int_sep_ratios.values():
                fallback_ratio = _coerce_ratio(v, "int_sep (fallback)")
                break
            for Nc in cluster_sizes_list:
                if Nc in int_sep_ratios:
                    ratio_map[Nc] = _coerce_ratio(int_sep_ratios[Nc], f"int_sep (Nc={Nc})")
                else:
                    if fallback_ratio is None:
                        raise ValueError(f"No int_sep ratio provided for Nc={Nc}.")
                    ratio_map[Nc] = fallback_ratio
        else:
            common_ratio = _coerce_ratio(int_sep_ratios, "int_sep")
            ratio_map = {Nc: common_ratio for Nc in cluster_sizes_list}

        # Generate per-(U, V) mu_0 arrays
        mu_arrays = {}
        for U in u_list:
            for V in v_list:
                half_range = max(mu_range_factor * (abs(U) + abs(V)), mu_min_range)
                mu_arrays[(U, V)] = np.linspace(-half_range, half_range, n_mu_points)

    # Storage structures
    cluster_fillings: Dict[Tuple[int, float, float], np.ndarray]
    cluster_energies: Dict[Tuple[int, float, float], np.ndarray]
    idmrg_fill_cache: Dict[Tuple[float, float], np.ndarray]
    finite_dmrg_fill_cache: Dict[Tuple[float, float], np.ndarray]
    failed_calculations: List[Dict] = []

    if results is not None:
        # Load from pre-computed results
        if isinstance(results, (str, os.PathLike)):
            results_path = Path(results)
            if not results_path.exists():
                raise ValueError(f"Results file not found: {results_path}")
            with open(results_path, 'rb') as fh:
                results = pickle.load(fh)
        elif not isinstance(results, dict):
            raise ValueError("results must be a dict or path-like object.")

        save_data = False
        print("Using precomputed results; skipping new simulations.")
        params = results.get('parameters', {})

        # Reconstruct mu_arrays
        stored_mu = results.get('mu_arrays', {})
        mu_arrays = {}
        for k, v in stored_mu.items():
            u_val, v_val = map(float, k.split('_'))
            mu_arrays[(u_val, v_val)] = np.asarray(v)

        # Reconstruct lists from stored data if not provided
        serialized_fills = results.get('cluster_fillings', {})
        serialized_energies = results.get('cluster_energies', {})
        if not cluster_sizes_list:
            cluster_sizes_list = sorted(int(k) for k in serialized_fills.keys())
        if not u_list:
            u_list = sorted({u for u, _ in mu_arrays.keys()})
        if not v_list:
            v_list = sorted({v for _, v in mu_arrays.keys()})

        # Rebuild ratio_map from stored data
        stored_ratios = results.get('int_sep_ratios', {})
        if stored_ratios and not ratio_map:
            ratio_map = {}
            for k, val in stored_ratios.items():
                ratio_map[int(k)] = _coerce_ratio(val, f"int_sep (Nc={k})")

        # Load cluster data
        cluster_fillings = {}
        cluster_energies = {}
        for Nc in cluster_sizes_list:
            for U in u_list:
                for V in v_list:
                    key = (Nc, U, V)
                    n_mu = len(mu_arrays[(U, V)])
                    uv_key = f"{U}_{V}"

                    fill_data = serialized_fills.get(str(Nc), {}).get(uv_key)
                    cluster_fillings[key] = np.asarray(fill_data, dtype=float) if fill_data is not None else np.full(n_mu, np.nan)

                    en_data = serialized_energies.get(str(Nc), {}).get(uv_key)
                    cluster_energies[key] = np.asarray(en_data, dtype=float) if en_data is not None else np.full(n_mu, np.nan)

        # Load DMRG caches
        idmrg_fill_cache = {}
        finite_dmrg_fill_cache = {}
        stored_idmrg = results.get('idmrg_fillings', {})
        stored_finite = results.get('finite_dmrg_fillings', {})
        for U in u_list:
            for V in v_list:
                uv_key = f"{U}_{V}"
                n_mu = len(mu_arrays[(U, V)])
                idmrg_fill_cache[(U, V)] = np.asarray(stored_idmrg.get(uv_key, []), dtype=float) if stored_idmrg.get(uv_key) is not None else np.full(n_mu, np.nan)
                finite_dmrg_fill_cache[(U, V)] = np.asarray(stored_finite.get(uv_key, []), dtype=float) if stored_finite.get(uv_key) is not None else np.full(n_mu, np.nan)

        # Override metadata
        v_sep_ratio = _coerce_ratio(params.get('v_sep_ratio', v_sep_ratio), "V separation")
        t = params.get('t', t)
        L = params.get('L', L)
        chi = params.get('chi', chi)
        states_retained = params.get('states_retained', states_retained)
        include_idmrg = params.get('include_idmrg', include_idmrg)
        include_finite_dmrg = params.get('include_finite_dmrg', include_finite_dmrg)
    else:
        # Initialize storage
        cluster_fillings = {
            (Nc, U, V): np.full(n_mu_points, np.nan, dtype=float)
            for Nc in cluster_sizes_list for U in u_list for V in v_list
        }
        cluster_energies = {
            (Nc, U, V): np.full(n_mu_points, np.nan, dtype=float)
            for Nc in cluster_sizes_list for U in u_list for V in v_list
        }
        idmrg_fill_cache = {
            (U, V): np.full(n_mu_points, np.nan, dtype=float)
            for U in u_list for V in v_list
        }
        finite_dmrg_fill_cache = {
            (U, V): np.full(n_mu_points, np.nan, dtype=float)
            for U in u_list for V in v_list
        }

        print("=" * 60)
        print("Running compressibility (cluster sizes) calculations")
        print(f"Cluster sizes: {cluster_sizes_list}")
        print(f"int_sep per Nc: {ratio_map}")
        print(f"v_sep = {format_sep_as_pi(v_sep_ratio)}")
        print(f"U values: {u_list}")
        print(f"V values: {v_list}")
        print(f"mu_0 points per (U, V): {n_mu_points}")
        print("=" * 60)

        # Compute cluster results
        for Nc in tqdm(cluster_sizes_list, desc="Cluster sizes", ncols=80):
            int_sep_ratio = ratio_map[Nc]
            for V in v_list:
                for U in u_list:
                    mu_arr = mu_arrays[(U, V)]
                    for mu_idx, mu_0 in enumerate(mu_arr):
                        physical_params = PhysicalParams(U=U, mu_0=mu_0, V=V, t=t)
                        run_config = ClusterModelConfig(
                            L=L,
                            int_cluster_size=Nc,
                            cluster_separation_ratio=int_sep_ratio,
                            V_separation_ratio=v_sep_ratio,
                            ham_lib='quspin',
                            physical_params=physical_params,
                            model_bc='periodic',
                            int_cluster_bc='periodic',
                            super_cluster_bc='periodic',
                            solver_method=solver_method,
                            states_retained=states_retained,
                        )
                        try:
                            system_expectations, _, _ = get_general_expectations(
                                run_config, mu_eff=mu_0, return_mu=True,
                            )
                            energy, filling, _ = system_expectations
                            key = (Nc, U, V)
                            cluster_fillings[key][mu_idx] = filling / L
                            cluster_energies[key][mu_idx] = (energy + mu_0 * filling) / L
                        except Exception as exc:
                            import traceback
                            failed_calculations.append({
                                'method': f'cluster Nc={Nc}',
                                'params': {'U': U, 'V': V, 'mu_0': mu_0},
                                'error': str(exc) or repr(exc),
                                'traceback': traceback.format_exc(),
                            })

        # DMRG reference sweeps
        print("\n" + "=" * 60)
        print("Computing DMRG references")
        print("=" * 60)

        for U in u_list:
            for V in v_list:
                mu_arr = mu_arrays[(U, V)]
                for mu_idx, mu_0 in enumerate(tqdm(mu_arr, desc=f"DMRG U={U} V={V}", ncols=80)):
                    if include_idmrg:
                        try:
                            _, filling_dmrg, _ = run_dmrg_method(
                                U, mu_0, V, v_sep_ratio, t, L, chi,
                            )
                            idmrg_fill_cache[(U, V)][mu_idx] = filling_dmrg
                        except Exception as exc:
                            import traceback
                            failed_calculations.append({
                                'method': 'iDMRG',
                                'params': {'U': U, 'V': V, 'mu_0': mu_0},
                                'error': str(exc),
                                'traceback': traceback.format_exc(),
                            })

                    if include_finite_dmrg:
                        try:
                            _, _, filling_finite = get_gnd(
                                L, chi, U, t, mu_0, V, v_sep_ratio,
                            )
                            finite_dmrg_fill_cache[(U, V)][mu_idx] = filling_finite
                        except Exception as exc:
                            import traceback
                            failed_calculations.append({
                                'method': 'Finite DMRG',
                                'params': {'U': U, 'V': V, 'mu_0': mu_0},
                                'error': str(exc),
                                'traceback': traceback.format_exc(),
                            })

    # --- COLOR MAPPING ---
    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf',
    ]
    nc_color_map = {Nc: color_palette[idx % len(color_palette)] for idx, Nc in enumerate(cluster_sizes_list)}

    # --- PLOTTING ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)

    v_sep_label = format_sep_as_pi(v_sep_ratio)

    # Set up row/col mapping
    if row_is_U:
        row_values: list = u_list
        col_values: list = v_list
    else:
        row_values = v_list
        col_values = u_list

    def _get_U_V(ri: int, ci: int) -> Tuple[float, float]:
        if row_is_U:
            return current_row_values[ri], current_col_values[ci]
        return current_col_values[ci], current_row_values[ri]

    def _row_label(val) -> str:
        return f"U={val}" if row_is_U else f"V={val}"

    def _col_label(val) -> str:
        return f"V={val}" if row_is_U else f"U={val}"

    n_rows_total = len(row_values)
    n_cols_total = len(col_values)
    n_cols_pp = min(cols_per_page, n_cols_total)
    n_rows_pp = min(rows_per_page, n_rows_total)
    n_col_pages = int(np.ceil(n_cols_total / n_cols_pp))
    n_row_pages = int(np.ceil(n_rows_total / n_rows_pp))
    n_pages = n_col_pages * n_row_pages

    figures: List[go.Figure] = []
    html_paths: List[str] = []

    for page_idx in range(n_pages):
        row_page = page_idx // n_col_pages
        col_page = page_idx % n_col_pages

        row_start = row_page * n_rows_pp
        row_end = min(row_start + n_rows_pp, n_rows_total)
        col_start = col_page * n_cols_pp
        col_end = min(col_start + n_cols_pp, n_cols_total)

        current_row_values = row_values[row_start:row_end]
        current_col_values = col_values[col_start:col_end]
        n_rows_this = len(current_row_values)
        n_cols_this = len(current_col_values)

        subplot_titles = []
        for rv in current_row_values:
            for cv in current_col_values:
                subplot_titles.append(f"{_row_label(rv)}, {_col_label(cv)}")

        fig = make_subplots(
            rows=n_rows_this,
            cols=n_cols_this,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.08,
            vertical_spacing=0.12 / max(n_rows_this - 1, 1) if n_rows_this > 1 else 0.12,
        )

        legend_name_map = {}
        legend_idx = 1
        for r in range(1, n_rows_this + 1):
            for c in range(1, n_cols_this + 1):
                legend_name_map[(r, c)] = 'legend' if legend_idx == 1 else f'legend{legend_idx}'
                legend_idx += 1

        any_trace = False
        for row_idx in range(n_rows_this):
            row = row_idx + 1
            for col_idx in range(n_cols_this):
                col = col_idx + 1
                U, V = _get_U_V(row_idx, col_idx)
                mu_arr = mu_arrays[(U, V)]

                # DMRG reference lines
                dmrg_color = '#DAA520'
                if include_idmrg and np.any(np.isfinite(idmrg_fill_cache[(U, V)])):
                    fig.add_trace(
                        go.Scatter(
                            x=mu_arr.tolist(), y=idmrg_fill_cache[(U, V)].tolist(),
                            mode='lines+markers', name="iDMRG",
                            legendgroup=f"iDMRG_{row}_{col}",
                            marker=dict(color=dmrg_color, size=10, symbol='diamond'),
                            line=dict(color=dmrg_color, width=2, dash='dot'),
                            showlegend=True,
                            hovertemplate="μ=%{x:.3g}<br>n_iDMRG=%{y:.4f}<extra></extra>",
                            legend=legend_name_map[(row, col)],
                        ), row=row, col=col,
                    )
                    any_trace = True

                if include_finite_dmrg and np.any(np.isfinite(finite_dmrg_fill_cache[(U, V)])):
                    fig.add_trace(
                        go.Scatter(
                            x=mu_arr.tolist(), y=finite_dmrg_fill_cache[(U, V)].tolist(),
                            mode='lines+markers', name="DMRG",
                            legendgroup=f"finite_dmrg_{row}_{col}",
                            marker=dict(color=dmrg_color, size=10, symbol='diamond-open'),
                            line=dict(color=dmrg_color, width=2, dash='dash'),
                            showlegend=True,
                            hovertemplate="μ=%{x:.3g}<br>n_DMRG=%{y:.4f}<extra></extra>",
                            legend=legend_name_map[(row, col)],
                        ), row=row, col=col,
                    )
                    any_trace = True

                # Cluster lines (one per Nc)
                for Nc in cluster_sizes_list:
                    key = (Nc, U, V)
                    fills = cluster_fillings[key]
                    trace_color = nc_color_map[Nc]
                    int_sep = ratio_map[Nc]
                    sep_label = format_sep_as_pi(int_sep)

                    xs = [mu_arr[i] for i in range(len(mu_arr)) if np.isfinite(fills[i])]
                    ys = [fills[i] for i in range(len(mu_arr)) if np.isfinite(fills[i])]
                    if not xs:
                        continue

                    fig.add_trace(
                        go.Scatter(
                            x=xs, y=ys,
                            mode='lines+markers',
                            name=f"Nc={Nc}, m={sep_label}",
                            legendgroup=f"Nc_{Nc}_{row}_{col}",
                            marker=dict(color=trace_color, size=6),
                            line=dict(color=trace_color, width=2),
                            showlegend=True,
                            hovertemplate=f"Nc={Nc}<br>μ=%{{x:.3g}}<br>n=%{{y:.4f}}<extra></extra>",
                            legend=legend_name_map[(row, col)],
                        ), row=row, col=col,
                    )
                    any_trace = True

        if not any_trace:
            continue

        # Layout
        annotation_text = f"v_sep={v_sep_label}, t={t}, L={L}, χ={chi}"
        page_suffix = f" (page {page_idx + 1})" if n_pages > 1 else ""

        legend_configs = {}
        for ri in range(n_rows_this):
            for ci in range(n_cols_this):
                lk = legend_name_map[(ri + 1, ci + 1)]
                x_start = ci / n_cols_this
                y_start = 1.0 - ri / n_rows_this
                legend_configs[lk] = dict(
                    x=x_start + 0.01,
                    y=y_start - 0.01,
                    xanchor='left', yanchor='top',
                    font=dict(size=9),
                    bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='rgba(200,200,200,0.5)',
                )

        fig.update_layout(
            title_text=f"Compressibility: Cluster Sizes{page_suffix}<br><sub>{annotation_text}</sub>",
            height=350 * n_rows_this + 100,
            width=400 * n_cols_this,
            margin=dict(b=80),
            **legend_configs,
        )
        fig.update_xaxes(title_text="μ₀")
        fig.update_yaxes(title_text="n (filling/site)")

        if save_html:
            page_tag = f"_page_{page_idx + 1}" if n_pages > 1 else ""
            html_name = f"{filename_prefix}_L{L}_chi{chi}_{timestamp}{page_tag}.html"
            html_path = os.path.join(output_dir, html_name)
            fig.write_html(html_path)
            html_paths.append(html_path)
            print(f"Saved figure to {html_path}")

        figures.append(fig)

    # --- MATPLOTLIB FIGURES ---
    mpl_paths: List[str] = []
    if save_pdf:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        plt.rcParams.update({
            'font.family': 'serif', 'font.size': 10,
            'axes.labelsize': 12, 'axes.titlesize': 11,
            'legend.fontsize': 7,
            'xtick.labelsize': 9, 'ytick.labelsize': 9,
            'lines.linewidth': 1.5, 'lines.markersize': 4,
            'axes.linewidth': 0.8, 'grid.linewidth': 0.4, 'grid.alpha': 0.3,
            'figure.dpi': 150, 'savefig.dpi': 300,
            'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
        })
        try:
            plt.rcParams.update({
                'text.usetex': True,
                'text.latex.preamble': r'\usepackage{amsmath}',
            })
            _test_fig, _test_ax = plt.subplots(1, 1, figsize=(1, 1))
            _test_ax.set_xlabel(r"$\mu$")
            _test_fig.savefig(os.path.join(output_dir, "_latex_test.pdf"))
            plt.close(_test_fig)
            os.remove(os.path.join(output_dir, "_latex_test.pdf"))
            use_tex = True
        except Exception:
            plt.rcParams['text.usetex'] = False
            use_tex = False

        def _pi_label(sep_str: str) -> str:
            if use_tex:
                return '$' + sep_str.replace('π', r'\pi') + '$'
            return sep_str

        mpl_n_rows = len(row_values)
        mpl_n_cols = len(col_values)
        v_sep_latex = v_sep_label.replace('π', r'\pi') if use_tex else v_sep_label

        # Generate filling and relative-error figures
        plot_modes = [False]
        if include_finite_dmrg:
            plot_modes.append(True)

        for is_rel_error in plot_modes:
            mode_tag = 'rel_error' if is_rel_error else 'filling'

            fig_width = 3.2 * mpl_n_cols + 0.6
            fig_height = 2.8 * mpl_n_rows + 0.8
            mpl_fig, axs = plt.subplots(
                mpl_n_rows, mpl_n_cols,
                figsize=(fig_width, fig_height),
                squeeze=False, sharex=False,
            )

            REL_ERR_REF_THRESHOLD = 0.02

            for ri in range(mpl_n_rows):
                for ci in range(mpl_n_cols):
                    if row_is_U:
                        U, V = u_list[ri], v_list[ci]
                    else:
                        U, V = u_list[ci], v_list[ri]
                    mu_arr = mu_arrays[(U, V)]
                    ax = axs[ri, ci]

                    # DMRG reference lines (only on filling plot)
                    dmrg_color = '#B8860B'
                    if not is_rel_error:
                        if include_idmrg and np.any(np.isfinite(idmrg_fill_cache[(U, V)])):
                            ax.plot(
                                mu_arr, idmrg_fill_cache[(U, V)],
                                color=dmrg_color, ls=':', lw=2.0,
                                marker='D', ms=5, markerfacecolor='none', markeredgewidth=0.8,
                                label='iDMRG', zorder=10,
                            )
                        if include_finite_dmrg and np.any(np.isfinite(finite_dmrg_fill_cache[(U, V)])):
                            ax.plot(
                                mu_arr, finite_dmrg_fill_cache[(U, V)],
                                color=dmrg_color, ls='--', lw=2.0,
                                marker='D', ms=5, markerfacecolor='none', markeredgewidth=0.8,
                                label='DMRG', zorder=10,
                            )

                    # Cluster lines
                    reference_series = finite_dmrg_fill_cache[(U, V)] if is_rel_error else None
                    for Nc in cluster_sizes_list:
                        key = (Nc, U, V)
                        fills = cluster_fillings[key]
                        trace_color = nc_color_map[Nc]
                        int_sep = ratio_map[Nc]
                        sep_label = format_sep_as_pi(int_sep)

                        xs, ys = [], []
                        for mu_idx, mu_0 in enumerate(mu_arr):
                            f_val = fills[mu_idx]
                            if not np.isfinite(f_val):
                                continue
                            if is_rel_error and reference_series is not None:
                                ref = reference_series[mu_idx]
                                if not (np.isfinite(ref) and abs(ref) > REL_ERR_REF_THRESHOLD):
                                    continue
                                xs.append(mu_0)
                                ys.append(np.abs(f_val - ref) / np.abs(ref))
                            else:
                                xs.append(mu_0)
                                ys.append(f_val)

                        if not xs:
                            continue

                        ax.plot(
                            xs, ys, color=trace_color, ls='-', lw=1.5,
                            marker='o', ms=3,
                            label=f'$N_c={Nc}$, m={_pi_label(sep_label)}' if use_tex else f'Nc={Nc}, m={sep_label}',
                            zorder=5,
                        )

                    # Formatting
                    ax.set_xlabel(r'$\mu_0$' if use_tex else 'mu_0')
                    if ci == 0:
                        row_label = (f'$U={U:g}$' if use_tex else f'U={U:g}') if row_is_U else (f'$V={V:g}$' if use_tex else f'V={V:g}')
                        if is_rel_error:
                            ylabel = r'Rel.\ error in $n$' if use_tex else 'Rel. error in n'
                        else:
                            ylabel = r'$n$ (filling/site)' if use_tex else 'n (filling/site)'
                        ax.set_ylabel(f'{row_label}\n{ylabel}')
                    if ri == 0:
                        col_title = (f'$V = {V:g}$' if use_tex else f'V = {V:g}') if row_is_U else (f'$U = {U:g}$' if use_tex else f'U = {U:g}')
                        ax.set_title(col_title)

                    ax.grid(True, ls='--', alpha=0.3)
                    if not is_rel_error:
                        ax.set_ylim(-0.05, 2.05)
                    margin = 0.05 * (mu_arr[-1] - mu_arr[0]) if len(mu_arr) > 1 else 0.5
                    ax.set_xlim(mu_arr[0] - margin, mu_arr[-1] + margin)
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

                    ax.legend(
                        loc='upper right', framealpha=0.85,
                        edgecolor='0.7', handlelength=1.5,
                        borderpad=0.3, labelspacing=0.25, fontsize=6,
                    )

            # Suptitle
            if is_rel_error:
                suptitle = (
                    f'Relative Error in Filling: $v_{{\\mathrm{{sep}}}}={v_sep_latex}$, $t={t}$, $L={L}$, $\\chi={chi}$'
                    if use_tex else
                    f'Relative Error in Filling: v_sep={v_sep_label}, t={t}, L={L}, chi={chi}'
                )
            else:
                suptitle = (
                    f'Compressibility (Cluster Sizes): $v_{{\\mathrm{{sep}}}}={v_sep_latex}$, $t={t}$, $L={L}$, $\\chi={chi}$'
                    if use_tex else
                    f'Compressibility (Cluster Sizes): v_sep={v_sep_label}, t={t}, L={L}, chi={chi}'
                )
            mpl_fig.suptitle(suptitle, fontsize=13, y=1.01)
            mpl_fig.tight_layout(rect=[0, 0, 0.96, 1.0])

            for ext in ('pdf', 'svg'):
                mpl_name = f"{filename_prefix}_{mode_tag}_L{L}_chi{chi}_{timestamp}.{ext}"
                mpl_path = os.path.join(output_dir, mpl_name)
                mpl_fig.savefig(mpl_path, format=ext)
                mpl_paths.append(mpl_path)
                print(f"Saved matplotlib figure to {mpl_path}")

            if show_plots:
                plt.show()
            else:
                plt.close(mpl_fig)

    # --- SERIALIZATION ---
    fig = figures[0] if figures else None
    saved_paths: Dict[str, Any] = {}
    if save_html:
        saved_paths['html_pages'] = html_paths
        if html_paths:
            saved_paths['html'] = html_paths[0]
    if mpl_paths:
        saved_paths['mpl_figures'] = mpl_paths

    cluster_fillings_serialized: Dict[str, Dict[str, List[float]]] = {}
    cluster_energies_serialized: Dict[str, Dict[str, List[float]]] = {}
    for Nc in cluster_sizes_list:
        cluster_fillings_serialized[str(Nc)] = {}
        cluster_energies_serialized[str(Nc)] = {}
        for U in u_list:
            for V in v_list:
                uv_key = f"{U}_{V}"
                key = (Nc, U, V)
                cluster_fillings_serialized[str(Nc)][uv_key] = cluster_fillings[key].tolist()
                cluster_energies_serialized[str(Nc)][uv_key] = cluster_energies[key].tolist()

    mu_arrays_serialized = {f"{U}_{V}": mu_arrays[(U, V)].tolist() for U in u_list for V in v_list}
    idmrg_serialized = {f"{U}_{V}": idmrg_fill_cache[(U, V)].tolist() for U in u_list for V in v_list}
    finite_dmrg_serialized = {f"{U}_{V}": finite_dmrg_fill_cache[(U, V)].tolist() for U in u_list for V in v_list}
    int_sep_map_serialized = {str(Nc): list(ratio) for Nc, ratio in ratio_map.items()}

    results_payload = {
        'cluster_sizes': cluster_sizes_list,
        'int_sep_ratios': int_sep_map_serialized,
        'U_values': u_list,
        'V_values': v_list,
        'mu_arrays': mu_arrays_serialized,
        'cluster_fillings': cluster_fillings_serialized,
        'cluster_energies': cluster_energies_serialized,
        'idmrg_fillings': idmrg_serialized,
        'finite_dmrg_fillings': finite_dmrg_serialized,
        'parameters': {
            'v_sep_ratio': v_sep_ratio,
            't': t, 'L': L, 'chi': chi,
            'solver_method': solver_method,
            'states_retained': states_retained,
            'include_idmrg': include_idmrg,
            'include_finite_dmrg': include_finite_dmrg,
            'n_mu_points': n_mu_points,
            'mu_range_factor': mu_range_factor,
            'mu_min_range': mu_min_range,
        },
        'artifacts': saved_paths,
        'failures': failed_calculations,
    }

    if save_data:
        pickle_name = f"{filename_prefix}_L{L}_chi{chi}_{timestamp}.pkl"
        pickle_path = os.path.join(output_dir, pickle_name)
        with open(pickle_path, 'wb') as fh:
            pickle.dump(results_payload, fh)
        saved_paths['pickle'] = pickle_path
        print(f"Saved data to {pickle_path}")

    if failed_calculations:
        print("\n" + "=" * 60)
        print("WARNING: Some calculations failed")
        print("=" * 60)
        for failure in failed_calculations:
            params_desc = ', '.join(f"{k}={v}" for k, v in failure['params'].items())
            print(f"{failure['method']}: {params_desc}")
            error_lines = failure['error'].splitlines()
            print(f"  Error: {error_lines[0] if error_lines else '(no error message)'}")

    if show_plots and fig is not None:
        for f in figures:
            f.show()

    return fig, results_payload


def compare_filling_cluster_sizes(
    v_sep_ratio: Tuple[int, int],
    int_sep_ratios: Union[Tuple[int, int], Dict[int, Tuple[int, int]]],
    cluster_sizes: Sequence[int],
    U_values: Sequence[float],
    V_values: Sequence[float],
    *,
    t: float = 1.0,
    L: int = 20,
    chi: int = 32,
    solver_method: str = 'dense_ED',
    states_retained: int = 4,
    output_dir: str = 'large_files/plots',
    show_plots: bool = True,
    save_html: bool = True,
    save_data: bool = True,
    filename_prefix: str = 'filling_cluster_size_comparison',
    log_yaxis: bool = True,
    include_idmrg: bool = True,
    include_finite_dmrg: bool = True,
    include_timing: bool = False,
    include_timing_plot: bool = False,
    plot_relative_error: Union[bool, str] = False,
    dmrg_fixed_filling: bool = True,
    compute_localization: bool = False,
    results: Optional[Union[Dict, str, os.PathLike]] = None,
    axes: Tuple[str, str] = ('U', 'V'),
    cols_per_page: int = 3,
    shared_yaxis: Optional[str] = None,
) -> Tuple[go.Figure, Dict]:
    """
    Compare half-filling and quarter-filling results across cluster sizes.

    Creates a 2xN grid where:
    - Rows: Half-filling (top, n=1) and Quarter-filling (bottom, n=0.5)
    - Columns: Different values of the subplot parameter (V by default)
    - X-axis: Values of the x-axis parameter (U by default)
    - Lines on each subplot: Different cluster sizes (N_c)
    - Y-axis: Ground state energy per site (or relative error if plot_relative_error=True)

    Args:
        v_sep_ratio: Ratio controlling the AA modulation for V.
        int_sep_ratios: Either a single (p, q) tuple applied to every cluster
            size or a dict mapping each N_c to its specific interaction
            separation ratio.
        cluster_sizes: Iterable of cluster sizes (N_c); each becomes a separate line.
        U_values: Iterable of U values.
        V_values: Iterable of V strengths.
        t: Hopping parameter.
        L: System size used for the cluster method.
        chi: Bond dimension for iDMRG.
        solver_method: Diagonalisation backend for the cluster Hamiltonian.
        states_retained: Number of states retained in the cluster solver.
        output_dir: Directory for saved artifacts.
        show_plots: Whether to open the generated Plotly figure.
        save_html: If True, save the interactive figure as HTML.
        save_data: If True, pickle the raw numerical results.
        filename_prefix: Prefix for saved artifact names.
        log_yaxis: Plot the y-axis on a log scale (for relative error plots).
        include_idmrg: Whether to include iDMRG reference calculations.
        include_finite_dmrg: Whether to include finite DMRG reference calculations.
        include_timing: Whether to record timing information.
        include_timing_plot: Whether to generate timing plots.
        plot_relative_error: Controls what is plotted on the y-axis:
            - False (default): Plot raw ground state energies
            - True: Plot relative error |E_cluster - E_DMRG| / |E_DMRG|
            - 'both': Generate both raw energy and relative error plots
        dmrg_fixed_filling: If True (default), finite DMRG uses canonical ensemble (fixed N)
            to match the target filling. Set to False for grand canonical DMRG.
        results: Pre-computed results dict or path to pickle file to load instead of computing.
        axes: Tuple of (x_axis_param, subplot_param). Default ('U', 'V') plots U on the
            x-axis with each subplot column a different V. Use ('V', 'U') to plot V on the
            x-axis with each subplot column a different U.
        cols_per_page: Maximum number of subplot columns per page (default: 3).
            Additional columns overflow to new pages.
        shared_yaxis: Controls shared y-axis scaling across subplots in the same row.
            None (default) uses independent scaling. 'row' shares the y-axis range
            across all pages for each row. 'page' shares the y-axis range within
            each page for each row.

    Returns:
        (figure, results_dict) - When plot_relative_error='both', results_dict contains
        'energy_figure' and 'error_figure' keys with the respective figures.
    """
    # Warn if compute_localization is True but iDMRG is disabled
    if compute_localization and not include_idmrg:
        warnings.warn(
            "compute_localization=True but include_idmrg=False. "
            "Correlation length requires iDMRG data. No correlation lengths will be computed.",
            UserWarning,
        )
        compute_localization = False  # Disable to avoid downstream issues

    # Handle 'both' mode by calling recursively with False and True
    if plot_relative_error == 'both':
        # First generate raw energy plots
        fig_energy, results_energy = compare_filling_cluster_sizes(
            v_sep_ratio=v_sep_ratio,
            int_sep_ratios=int_sep_ratios,
            cluster_sizes=cluster_sizes,
            U_values=U_values,
            V_values=V_values,
            t=t,
            L=L,
            chi=chi,
            solver_method=solver_method,
            states_retained=states_retained,
            output_dir=output_dir,
            show_plots=False,  # Don't show yet
            save_html=save_html,
            save_data=save_data,
            filename_prefix=f"{filename_prefix}_energy",
            log_yaxis=log_yaxis,
            include_idmrg=include_idmrg,
            include_finite_dmrg=include_finite_dmrg,
            include_timing=include_timing,
            include_timing_plot=include_timing_plot,
            plot_relative_error=False,
            dmrg_fixed_filling=dmrg_fixed_filling,
            results=results,
            axes=axes,
            cols_per_page=cols_per_page,
            shared_yaxis=shared_yaxis,
        )

        # Then generate relative error plots, reusing computed results
        fig_error, results_error = compare_filling_cluster_sizes(
            v_sep_ratio=v_sep_ratio,
            int_sep_ratios=int_sep_ratios,
            cluster_sizes=cluster_sizes,
            U_values=U_values,
            V_values=V_values,
            t=t,
            L=L,
            chi=chi,
            solver_method=solver_method,
            states_retained=states_retained,
            output_dir=output_dir,
            show_plots=False,  # Don't show yet
            save_html=save_html,
            save_data=False,  # Data already saved from energy call
            filename_prefix=f"{filename_prefix}_error",
            log_yaxis=log_yaxis,
            include_idmrg=include_idmrg,
            include_finite_dmrg=include_finite_dmrg,
            include_timing=False,  # Already recorded from energy call
            include_timing_plot=False,
            plot_relative_error=True,
            dmrg_fixed_filling=dmrg_fixed_filling,
            results=results_energy,  # Reuse computed results
            axes=axes,
            cols_per_page=cols_per_page,
            shared_yaxis=shared_yaxis,
        )

        # Combine results
        combined_results = results_energy.copy()
        combined_results['energy_figure'] = fig_energy
        combined_results['error_figure'] = fig_error
        combined_results['artifacts']['error_html_pages'] = results_error.get('artifacts', {}).get('html_pages', [])
        if results_error.get('artifacts', {}).get('html'):
            combined_results['artifacts']['error_html'] = results_error['artifacts']['html']

        if show_plots:
            if fig_energy:
                fig_energy.show()
            if fig_error:
                fig_error.show()

        return fig_energy, combined_results

    if t is None:
        raise ValueError("Parameter t must be specified for the cluster calculations.")

    # Validate axes parameter
    if (not isinstance(axes, (tuple, list)) or len(axes) != 2
            or set(axes) != {'U', 'V'}):
        raise ValueError(f"axes must be ('U', 'V') or ('V', 'U'), got {axes!r}")
    x_param = axes[0]

    # Validate shared_yaxis parameter
    if shared_yaxis is not None and shared_yaxis not in ('row', 'page'):
        raise ValueError(f"shared_yaxis must be None, 'row', or 'page', got {shared_yaxis!r}")

    def _coerce_ratio(value, label: str) -> Tuple[int, int]:
        if value is None:
            raise ValueError(f"{label} ratio must be provided.")
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{label} ratio must be a length-2 iterable, got {value!r}.")
        try:
            p = int(round(value[0]))
            q = int(round(value[1]))
        except Exception as exc:
            raise ValueError(f"Could not parse {label} ratio {value!r} into integers.") from exc
        if q == 0:
            raise ValueError(f"Denominator for {label} ratio cannot be zero.")
        return (p, q)

    cluster_sizes = sorted({int(size) for size in cluster_sizes})
    if not cluster_sizes:
        raise ValueError("Provide at least one cluster size (N_c).")

    U_values = np.asarray(U_values, dtype=float)
    V_values = np.asarray(V_values, dtype=float)
    if U_values.ndim != 1 or U_values.size == 0:
        raise ValueError("U_values must be a 1-D array with at least one entry.")
    if V_values.ndim != 1 or V_values.size == 0:
        raise ValueError("V_values must be a 1-D array with at least one entry.")

    u_list = [float(u) for u in U_values]
    v_list = [float(v) for v in V_values]

    v_sep_ratio = _coerce_ratio(v_sep_ratio, "V separation")

    # Build ratio_map for int_sep_ratios
    if isinstance(int_sep_ratios, dict):
        ratio_map: Dict[int, Tuple[int, int]] = {}
        fallback_ratio: Optional[Tuple[int, int]] = None
        for v in int_sep_ratios.values():
            fallback_ratio = _coerce_ratio(v, "int_sep (fallback)")
            break
        for Nc in cluster_sizes:
            if Nc in int_sep_ratios:
                ratio_map[Nc] = _coerce_ratio(int_sep_ratios[Nc], f"int_sep (Nc={Nc})")
            else:
                if fallback_ratio is None:
                    raise ValueError(f"No int_sep ratio provided for cluster size Nc={Nc}.")
                warnings.warn(f"No int_sep ratio provided for Nc={Nc}; using fallback {fallback_ratio}.")
                ratio_map[Nc] = fallback_ratio
    else:
        common_ratio = _coerce_ratio(int_sep_ratios, "int_sep")
        ratio_map = {Nc: common_ratio for Nc in cluster_sizes}

    # Define the two filling modes
    filling_modes = ['half', 'quarter']
    filling_targets = {'half': 1.0, 'quarter': 0.5}

    # Storage structures
    # Key: (Nc, V, filling_mode) -> energy value indexed by U
    cluster_results: Dict[Tuple[int, float, str], np.ndarray]
    cluster_fillings: Dict[Tuple[int, float, str], np.ndarray]
    # DMRG caches: indexed by (u_idx, v_idx, filling_mode_idx)
    idmrg_cache: np.ndarray
    finite_dmrg_cache: np.ndarray
    idmrg_fill_cache: np.ndarray
    finite_dmrg_fill_cache: np.ndarray
    failed_calculations: List[Dict] = []
    timing_csv_path = os.environ.get("TIMING_CSV") if include_timing else None
    timing_recorder = TimingRecorder(csv_path=timing_csv_path) if include_timing else None

    if results is not None:
        # Load from pre-computed results
        if isinstance(results, (str, os.PathLike)):
            results_path = Path(results)
            if not results_path.exists():
                raise ValueError(f"Results file not found: {results_path}")
            with open(results_path, 'rb') as fh:
                results = pickle.load(fh)
        elif not isinstance(results, dict):
            raise ValueError("results must be a dict or path-like object when provided.")

        save_data = False
        print("Using precomputed results payload; skipping new simulations.")
        params = results.get('parameters', {})

        # Override U/V values and cluster sizes from stored results
        stored_u = results.get('U_values')
        if stored_u is not None:
            u_list = [float(u) for u in stored_u]
        stored_v = results.get('V_values')
        if stored_v is not None:
            v_list = [float(v) for v in stored_v]
        stored_cs = results.get('cluster_sizes')
        if stored_cs is not None:
            cluster_sizes = sorted({int(s) for s in stored_cs})

        # Load caches
        idmrg_cache = np.asarray(results.get('idmrg_energies', []), dtype=float)
        if idmrg_cache.size == 0:
            idmrg_cache = np.full((len(u_list), len(v_list), 2), np.nan, dtype=float)
        finite_dmrg_cache = np.asarray(results.get('finite_dmrg_energies', []), dtype=float)
        if finite_dmrg_cache.size == 0:
            finite_dmrg_cache = np.full((len(u_list), len(v_list), 2), np.nan, dtype=float)
        idmrg_fill_cache = np.asarray(results.get('idmrg_fillings', []), dtype=float)
        if idmrg_fill_cache.size == 0:
            idmrg_fill_cache = np.full((len(u_list), len(v_list), 2), np.nan, dtype=float)
        finite_dmrg_fill_cache = np.asarray(results.get('finite_dmrg_fillings', []), dtype=float)
        if finite_dmrg_fill_cache.size == 0:
            finite_dmrg_fill_cache = np.full((len(u_list), len(v_list), 2), np.nan, dtype=float)

        # Load correlation length cache if available
        stored_corr_length = results.get('idmrg_correlation_lengths')
        if stored_corr_length is not None:
            idmrg_corr_length_cache = np.asarray(stored_corr_length, dtype=float)
        else:
            idmrg_corr_length_cache = np.full((len(u_list), len(v_list), 2), np.nan, dtype=float) if compute_localization else None

        # Load cluster results
        serialized_clusters = results.get('cluster_energies', {})
        serialized_fills = results.get('cluster_fillings', {})
        cluster_results = {}
        cluster_fillings = {}

        for Nc in cluster_sizes:
            for V in v_list:
                v_key = str(V)
                for fill_idx, fill_mode in enumerate(filling_modes):
                    key = (Nc, V, fill_mode)
                    cluster_by_nc = serialized_clusters.get(str(Nc), {})
                    cluster_by_v = cluster_by_nc.get(v_key, {})
                    series = cluster_by_v.get(fill_mode)
                    if series is None:
                        cluster_results[key] = np.full(len(u_list), np.nan, dtype=float)
                    else:
                        cluster_results[key] = np.asarray(series, dtype=float)

                    fill_by_nc = serialized_fills.get(str(Nc), {})
                    fill_by_v = fill_by_nc.get(v_key, {})
                    fill_series = fill_by_v.get(fill_mode)
                    if fill_series is None:
                        cluster_fillings[key] = np.full(len(u_list), np.nan, dtype=float)
                    else:
                        cluster_fillings[key] = np.asarray(fill_series, dtype=float)

        # Override metadata from results where available
        stored_ratio_map = results.get('int_sep_ratios')
        if stored_ratio_map:
            converted_ratio_map = {}
            for k, val in stored_ratio_map.items():
                try:
                    Nc_key = int(k)
                except (TypeError, ValueError):
                    Nc_key = k
                converted_ratio_map[Nc_key] = _coerce_ratio(val, f"int_sep (Nc={Nc_key})")
            ratio_map = converted_ratio_map
        params_v_sep = params.get('v_sep_ratio', v_sep_ratio)
        v_sep_ratio = _coerce_ratio(params_v_sep, "V separation")
        t = params.get('t', t)
        L = params.get('L', L)
        chi = params.get('chi', chi)
        states_retained = params.get('states_retained', states_retained)
    else:
        # Initialize storage
        cluster_results = {
            (Nc, V, fill_mode): np.full(len(u_list), np.nan, dtype=float)
            for Nc in cluster_sizes
            for V in v_list
            for fill_mode in filling_modes
        }
        cluster_fillings = {
            (Nc, V, fill_mode): np.full(len(u_list), np.nan, dtype=float)
            for Nc in cluster_sizes
            for V in v_list
            for fill_mode in filling_modes
        }
        # DMRG caches: shape (n_U, n_V, 2) for half/quarter filling
        idmrg_cache = np.full((len(u_list), len(v_list), 2), np.nan, dtype=float)
        finite_dmrg_cache = np.full((len(u_list), len(v_list), 2), np.nan, dtype=float)
        idmrg_fill_cache = np.full((len(u_list), len(v_list), 2), np.nan, dtype=float)
        finite_dmrg_fill_cache = np.full((len(u_list), len(v_list), 2), np.nan, dtype=float)
        # Correlation length cache (only for iDMRG)
        idmrg_corr_length_cache = np.full((len(u_list), len(v_list), 2), np.nan, dtype=float) if compute_localization else None

        # Compute DMRG references
        print("=" * 60)
        print("Computing DMRG reference energies for half and quarter filling")
        print("=" * 60)

        for fill_idx, fill_mode in enumerate(filling_modes):
            target_fill = filling_targets[fill_mode]
            print(f"\n--- {fill_mode.capitalize()}-filling (n={target_fill}) ---")

            for u_idx, U in enumerate(tqdm(u_list, desc=f"DMRG {fill_mode}", ncols=80)):
                for v_idx, V in enumerate(v_list):
                    # iDMRG (grand canonical with mu = U/2 for half-filling, mu=0 for quarter)
                    if include_idmrg:
                        try:
                            mu_0 = U / 2.0 if fill_mode == 'half' else 0.0
                            meta = {
                                "method": "iDMRG",
                                "U": U, "V": V, "t": t, "L": L,
                                "Nc": None, "int_sep": None, "v_sep": v_sep_ratio,
                                "filling_mode": fill_mode,
                            }
                            if compute_localization:
                                energy_dmrg, filling_dmrg, psi, localization = time_call(
                                    timing_recorder, meta,
                                    run_dmrg_method, U, mu_0, V, v_sep_ratio, t, L, chi,
                                    compute_localization=True,
                                )
                                xi = localization.get('correlation_length') if localization else None
                                if xi is not None and idmrg_corr_length_cache is not None:
                                    idmrg_corr_length_cache[u_idx, v_idx, fill_idx] = xi
                            else:
                                energy_dmrg, filling_dmrg, _ = time_call(
                                    timing_recorder, meta,
                                    run_dmrg_method, U, mu_0, V, v_sep_ratio, t, L, chi,
                                )
                            dmrg_value = energy_dmrg + mu_0 * filling_dmrg
                            idmrg_cache[u_idx, v_idx, fill_idx] = dmrg_value
                            idmrg_fill_cache[u_idx, v_idx, fill_idx] = filling_dmrg
                        except Exception as exc:
                            import traceback
                            error_msg = str(exc) or f"{type(exc).__name__}: {repr(exc)}"
                            failed_calculations.append({
                                'method': f'iDMRG ({fill_mode})',
                                'params': {'U': U, 'V': V},
                                'error': error_msg,
                                'traceback': traceback.format_exc(),
                            })

                    # Finite DMRG
                    if include_finite_dmrg:
                        try:
                            if dmrg_fixed_filling:
                                # Fixed N DMRG
                                meta = {
                                    "method": "DMRG_fixed_N",
                                    "U": U, "V": V, "t": t, "L": L,
                                    "filling_target": target_fill,
                                    "filling_mode": fill_mode,
                                }
                                energy_finite, _, filling_finite = time_call(
                                    timing_recorder, meta,
                                    get_gnd_fixed_filling, L, chi, target_fill, U, t, V, v_sep_ratio,
                                )
                            else:
                                # Grand canonical
                                mu_0 = U / 2.0 if fill_mode == 'half' else 0.0
                                meta = {
                                    "method": "DMRG",
                                    "U": U, "V": V, "t": t, "L": L,
                                    "mu_0": mu_0,
                                    "filling_mode": fill_mode,
                                }
                                energy_finite, _, filling_finite = time_call(
                                    timing_recorder, meta,
                                    get_gnd, L, chi, U, t, mu_0, V, v_sep_ratio,
                                )
                            energy_finite_per_site = energy_finite / L
                            finite_dmrg_cache[u_idx, v_idx, fill_idx] = energy_finite_per_site
                            finite_dmrg_fill_cache[u_idx, v_idx, fill_idx] = filling_finite
                        except Exception as exc:
                            import traceback
                            error_msg = str(exc) or f"{type(exc).__name__}: {repr(exc)}"
                            failed_calculations.append({
                                'method': f'Finite DMRG ({fill_mode})',
                                'params': {'U': U, 'V': V},
                                'error': error_msg,
                                'traceback': traceback.format_exc(),
                            })

        # Compute cluster results
        print("\n" + "=" * 60)
        print("Running cluster calculations")
        print("=" * 60)

        for fill_idx, fill_mode in enumerate(filling_modes):
            target_fill = filling_targets[fill_mode]
            print(f"\n--- {fill_mode.capitalize()}-filling (n={target_fill}) ---")

            for Nc in tqdm(cluster_sizes, desc=f"Cluster {fill_mode}", ncols=80):
                int_sep_ratio = ratio_map[Nc]
                for v_idx, V in enumerate(v_list):
                    for u_idx, U in enumerate(u_list):
                        physical_params = PhysicalParams(U=U, mu_0=0.0, V=V, t=t)
                        run_config = ClusterModelConfig(
                            L=L,
                            int_cluster_size=Nc,
                            cluster_separation_ratio=int_sep_ratio,
                            V_separation_ratio=v_sep_ratio,
                            ham_lib='quspin',
                            physical_params=physical_params,
                            model_bc='periodic',
                            int_cluster_bc='periodic',
                            super_cluster_bc='periodic',
                            solver_method=solver_method,
                            states_retained=states_retained,
                        )
                        try:
                            meta = {
                                "method": "cluster_ED",
                                "U": U, "V": V, "t": t, "L": L,
                                "Nc": Nc, "int_sep": int_sep_ratio, "v_sep": v_sep_ratio,
                                "filling_mode": fill_mode,
                            }
                            system_expectations, _ = time_call(
                                timing_recorder, meta,
                                get_general_expectations, run_config, timing_recorder=timing_recorder,
                                set_filling=target_fill,
                            )
                            energy, filling, mu_eff = system_expectations
                            energy_per_site = energy / L
                            cluster_results[(Nc, V, fill_mode)][u_idx] = energy_per_site
                            cluster_fillings[(Nc, V, fill_mode)][u_idx] = filling / L
                        except Exception as exc:
                            import traceback
                            error_msg = str(exc) or f"{type(exc).__name__}: {repr(exc)}"
                            failed_calculations.append({
                                'method': f'Cluster ({fill_mode})',
                                'params': {'U': U, 'V': V, 'Nc': Nc},
                                'error': error_msg,
                                'traceback': traceback.format_exc(),
                            })

    # --- PLOTTING LOGIC ---
    # Color palette for cluster sizes (N_c)
    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf',
    ]
    nc_color_map = {Nc: color_palette[idx % len(color_palette)] for idx, Nc in enumerate(cluster_sizes)}

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)

    # Determine axis configuration for plotting
    if x_param == 'U':
        x_list_plot = u_list       # x-axis values
        subplot_list = v_list      # one subplot column per value
        x_label = 'U'
        subplot_label = 'V'
    else:
        x_list_plot = v_list
        subplot_list = u_list
        x_label = 'V'
        subplot_label = 'U'

    # Pre-compute shared y-axis ranges per row (filling mode) if requested
    energy_y_ranges: Dict[int, Optional[List[float]]] = {0: None, 1: None}
    fill_y_ranges: Dict[int, Optional[List[float]]] = {0: None, 1: None}
    if shared_yaxis == 'row':
        for fill_idx, fill_mode in enumerate(filling_modes):
            all_energy_ys: List[float] = []
            all_fill_ys: List[float] = []
            for sp_val in subplot_list:
                sp_full_idx = subplot_list.index(sp_val)
                for x_idx in range(len(x_list_plot)):
                    if x_param == 'U':
                        u_idx, v_idx = x_idx, sp_full_idx
                        V_key = sp_val
                    else:
                        u_idx, v_idx = sp_full_idx, x_idx
                        V_key = x_list_plot[x_idx]

                    # DMRG energies
                    for cache in [idmrg_cache, finite_dmrg_cache]:
                        val = cache[u_idx, v_idx, fill_idx]
                        if np.isfinite(val):
                            if plot_relative_error:
                                pass  # handled below with cluster
                            else:
                                all_energy_ys.append(val)

                    # Cluster energies
                    for Nc in cluster_sizes:
                        key = (Nc, V_key, fill_mode)
                        c_en = cluster_results[key][u_idx]
                        if np.isfinite(c_en):
                            if plot_relative_error:
                                ref = finite_dmrg_cache[u_idx, v_idx, fill_idx]
                                if np.isfinite(ref) and ref != 0:
                                    all_energy_ys.append(np.abs(c_en - ref) / np.abs(ref))
                            else:
                                all_energy_ys.append(c_en)

                    # Fillings
                    dmrg_fill = finite_dmrg_fill_cache[u_idx, v_idx, fill_idx]
                    if np.isfinite(dmrg_fill):
                        all_fill_ys.append(dmrg_fill)
                    for Nc in cluster_sizes:
                        key = (Nc, V_key, fill_mode)
                        c_fill = cluster_fillings[key][u_idx]
                        if np.isfinite(c_fill):
                            all_fill_ys.append(c_fill)

            if all_energy_ys:
                margin = 0.05 * (max(all_energy_ys) - min(all_energy_ys)) if max(all_energy_ys) != min(all_energy_ys) else 0.1
                energy_y_ranges[fill_idx] = [min(all_energy_ys) - margin, max(all_energy_ys) + margin]
            if all_fill_ys:
                margin = 0.05 * (max(all_fill_ys) - min(all_fill_ys)) if max(all_fill_ys) != min(all_fill_ys) else 0.1
                fill_y_ranges[fill_idx] = [min(all_fill_ys) - margin, max(all_fill_ys) + margin]

    # Create figure: 2 rows (half/quarter) x N_subplot columns
    n_cols = len(subplot_list)
    n_cols_per_page = cols_per_page
    n_pages = int(np.ceil(n_cols / n_cols_per_page))

    figures: List[go.Figure] = []
    html_paths: List[str] = []
    filling_figures: List[go.Figure] = []
    filling_html_paths: List[str] = []
    corr_length_figures: List[go.Figure] = []
    corr_length_html_paths: List[str] = []

    for page_idx in range(n_pages):
        start_col = page_idx * n_cols_per_page
        end_col = min(start_col + n_cols_per_page, n_cols)
        current_subplot_values = subplot_list[start_col:end_col]
        n_cols_this_page = len(current_subplot_values)

        # Per-page y-axis range computation (overrides global ranges each page)
        if shared_yaxis == 'page':
            energy_y_ranges = {0: None, 1: None}
            fill_y_ranges = {0: None, 1: None}
            for fill_idx_p, fill_mode_p in enumerate(filling_modes):
                page_energy_ys: List[float] = []
                page_fill_ys: List[float] = []
                for sp_val_p in current_subplot_values:
                    sp_idx_p = subplot_list.index(sp_val_p)
                    for x_idx_p in range(len(x_list_plot)):
                        if x_param == 'U':
                            u_idx_p, v_idx_p = x_idx_p, sp_idx_p
                            V_key_p = sp_val_p
                        else:
                            u_idx_p, v_idx_p = sp_idx_p, x_idx_p
                            V_key_p = x_list_plot[x_idx_p]

                        for cache_p in [idmrg_cache, finite_dmrg_cache]:
                            val_p = cache_p[u_idx_p, v_idx_p, fill_idx_p]
                            if np.isfinite(val_p):
                                if not plot_relative_error:
                                    page_energy_ys.append(val_p)

                        for Nc_p in cluster_sizes:
                            key_p = (Nc_p, V_key_p, fill_mode_p)
                            c_en_p = cluster_results[key_p][u_idx_p]
                            if np.isfinite(c_en_p):
                                if plot_relative_error:
                                    ref_p = finite_dmrg_cache[u_idx_p, v_idx_p, fill_idx_p]
                                    if np.isfinite(ref_p) and ref_p != 0:
                                        page_energy_ys.append(np.abs(c_en_p - ref_p) / np.abs(ref_p))
                                else:
                                    page_energy_ys.append(c_en_p)

                        dmrg_fill_p = finite_dmrg_fill_cache[u_idx_p, v_idx_p, fill_idx_p]
                        if np.isfinite(dmrg_fill_p):
                            page_fill_ys.append(dmrg_fill_p)
                        for Nc_p in cluster_sizes:
                            key_p = (Nc_p, V_key_p, fill_mode_p)
                            c_fill_p = cluster_fillings[key_p][u_idx_p]
                            if np.isfinite(c_fill_p):
                                page_fill_ys.append(c_fill_p)

                if page_energy_ys:
                    margin_e = 0.05 * (max(page_energy_ys) - min(page_energy_ys)) if max(page_energy_ys) != min(page_energy_ys) else 0.1
                    energy_y_ranges[fill_idx_p] = [min(page_energy_ys) - margin_e, max(page_energy_ys) + margin_e]
                if page_fill_ys:
                    margin_f = 0.05 * (max(page_fill_ys) - min(page_fill_ys)) if max(page_fill_ys) != min(page_fill_ys) else 0.1
                    fill_y_ranges[fill_idx_p] = [min(page_fill_ys) - margin_f, max(page_fill_ys) + margin_f]

        # Subplot titles
        top_titles = [f"Half-filling (n=1): {subplot_label}={sv}" for sv in current_subplot_values]
        bottom_titles = [f"Quarter-filling (n=0.5): {subplot_label}={sv}" for sv in current_subplot_values]

        fig = make_subplots(
            rows=2,
            cols=n_cols_this_page,
            subplot_titles=top_titles + bottom_titles,
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )

        # Build legend name mapping
        legend_name_map = {}
        legend_idx = 1
        for r in range(1, 3):
            for c in range(1, n_cols_this_page + 1):
                if legend_idx == 1:
                    legend_name_map[(r, c)] = 'legend'
                else:
                    legend_name_map[(r, c)] = f'legend{legend_idx}'
                legend_idx += 1

        any_trace = False
        for col_idx, sp_val in enumerate(current_subplot_values):
            col = col_idx + 1
            sp_full_idx = subplot_list.index(sp_val)

            for row_idx, fill_mode in enumerate(filling_modes):
                row = row_idx + 1
                fill_idx = row_idx

                # Add DMRG reference line (gold color, dotted, big diamonds)
                dmrg_color = '#DAA520'  # Gold

                # iDMRG reference line
                if include_idmrg and not plot_relative_error:
                    idmrg_xs, idmrg_ys = [], []
                    for x_idx, x_val in enumerate(x_list_plot):
                        if x_param == 'U':
                            u_idx, v_idx = x_idx, sp_full_idx
                        else:
                            u_idx, v_idx = sp_full_idx, x_idx
                        idmrg_energy = idmrg_cache[u_idx, v_idx, fill_idx]
                        if np.isfinite(idmrg_energy):
                            idmrg_xs.append(x_val)
                            idmrg_ys.append(idmrg_energy)
                    if idmrg_xs:
                        fig.add_trace(
                            go.Scatter(
                                x=idmrg_xs,
                                y=idmrg_ys,
                                mode='lines',
                                name="iDMRG",
                                legendgroup=f"iDMRG_{row}_{col}",
                                line=dict(color=dmrg_color, width=2, dash='dot'),
                                showlegend=True,
                                hovertemplate=f"iDMRG<br>{x_label}=%{{x}}<br>E=%{{y:.6f}}<extra></extra>",
                                legend=legend_name_map[(row, col)],
                            ),
                            row=row,
                            col=col,
                        )
                        any_trace = True

                # Finite DMRG reference line
                if include_finite_dmrg and not plot_relative_error:
                    dmrg_xs, dmrg_ys = [], []
                    for x_idx, x_val in enumerate(x_list_plot):
                        if x_param == 'U':
                            u_idx, v_idx = x_idx, sp_full_idx
                        else:
                            u_idx, v_idx = sp_full_idx, x_idx
                        dmrg_energy = finite_dmrg_cache[u_idx, v_idx, fill_idx]
                        if np.isfinite(dmrg_energy):
                            dmrg_xs.append(x_val)
                            dmrg_ys.append(dmrg_energy)
                    if dmrg_xs:
                        fig.add_trace(
                            go.Scatter(
                                x=dmrg_xs,
                                y=dmrg_ys,
                                mode='lines+markers',
                                name="DMRG",
                                legendgroup=f"DMRG_{row}_{col}",
                                marker=dict(color=dmrg_color, size=10, symbol='diamond'),
                                line=dict(color=dmrg_color, width=2, dash='dot'),
                                showlegend=True,
                                hovertemplate=f"DMRG<br>{x_label}=%{{x}}<br>E=%{{y:.6f}}<extra></extra>",
                                legend=legend_name_map[(row, col)],
                            ),
                            row=row,
                            col=col,
                        )
                        any_trace = True

                # Add cluster lines for each N_c value
                for nc_idx, Nc in enumerate(cluster_sizes):
                    xs, ys, hover_text = [], [], []
                    for x_idx, x_val in enumerate(x_list_plot):
                        if x_param == 'U':
                            u_idx, v_idx = x_idx, sp_full_idx
                            V_key, U_val = sp_val, x_val
                        else:
                            u_idx, v_idx = sp_full_idx, x_idx
                            V_key, U_val = x_val, sp_val
                        key = (Nc, V_key, fill_mode)
                        c_en = cluster_results[key][u_idx]
                        if not np.isfinite(c_en):
                            continue

                        if plot_relative_error:
                            ref = finite_dmrg_cache[u_idx, v_idx, fill_idx]
                            if not (np.isfinite(ref) and ref != 0):
                                continue
                            c_err = np.abs(c_en - ref) / np.abs(ref)
                            xs.append(x_val)
                            ys.append(c_err)
                            hover_text.append(
                                f"U={U_val}<br>V={V_key}<br>N_c={Nc}<br>rel_err={c_err:.2%}"
                            )
                        else:
                            xs.append(x_val)
                            ys.append(c_en)
                            hover_text.append(
                                f"U={U_val}<br>V={V_key}<br>N_c={Nc}<br>E={c_en:.6f}"
                            )

                    if not xs:
                        continue

                    trace_color = nc_color_map[Nc]
                    fig.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode='lines+markers',
                            name=f"N_c={Nc}",
                            legendgroup=f"Nc_{Nc}_{row}_{col}",
                            marker=dict(color=trace_color, size=8),
                            line=dict(color=trace_color, width=2),
                            showlegend=True,
                            hovertemplate="%{text}<extra></extra>",
                            text=hover_text,
                            legend=legend_name_map[(row, col)],
                        ),
                        row=row,
                        col=col,
                    )
                    any_trace = True

                # Axis labels
                fig.update_xaxes(title_text=x_label, row=row, col=col)
                y_title = "Relative error" if plot_relative_error else "Energy per site"
                y_kwargs = dict(
                    title_text=y_title,
                    row=row,
                    col=col,
                    type='log' if (plot_relative_error and log_yaxis) else 'linear',
                )
                if plot_relative_error:
                    y_kwargs['tickformat'] = '.1%'
                if shared_yaxis is not None and energy_y_ranges[fill_idx] is not None:
                    y_range = energy_y_ranges[fill_idx]
                    if plot_relative_error and log_yaxis:
                        # Log-scale axis needs range in log10 space
                        lo = max(y_range[0], 1e-15)  # avoid log(0)
                        hi = max(y_range[1], 1e-15)
                        y_kwargs['range'] = [np.log10(lo), np.log10(hi)]
                    else:
                        y_kwargs['range'] = y_range
                fig.update_yaxes(**y_kwargs)

        if not any_trace:
            raise RuntimeError("No valid data points available to plot.")

        # Layout and annotations
        v_sep_label = format_sep_as_pi(v_sep_ratio)
        annotation_parts = [
            f"v_sep={v_sep_label}",
            f"t={t}",
            f"L={L}",
            f"chi={chi}",
            f"states={states_retained}",
        ]
        annotation_text = ", ".join(annotation_parts)
        if n_pages > 1:
            annotation_text += f" | Page {page_idx + 1}/{n_pages}"

        # Calculate subplot domains for legend positioning
        h_spacing = 0.08
        v_spacing = 0.12
        subplot_width = (1.0 - h_spacing * (n_cols_this_page - 1)) / n_cols_this_page
        subplot_height = (1.0 - v_spacing) / 2

        legend_configs = {}
        for col_idx_leg in range(n_cols_this_page):
            for row_idx_leg in range(2):
                row_leg = row_idx_leg + 1
                col_leg = col_idx_leg + 1
                legend_key = legend_name_map[(row_leg, col_leg)]

                x_end = col_idx_leg * (subplot_width + h_spacing) + subplot_width
                y_start = 1.0 - row_idx_leg * (subplot_height + v_spacing) - subplot_height

                legend_configs[legend_key] = dict(
                    x=x_end - 0.01,
                    y=y_start + 0.02,
                    xanchor='right',
                    yanchor='bottom',
                    bgcolor='rgba(255, 255, 255, 0.85)',
                    bordercolor='rgba(0, 0, 0, 0.3)',
                    borderwidth=1,
                    font=dict(size=8),
                    itemsizing='constant',
                    tracegroupgap=0,
                    itemwidth=30,
                )

        fig.update_layout(
            title=dict(
                text=f"Relative Error vs {x_label} (Half vs Quarter Filling)" if plot_relative_error
                     else f"Ground State Energy vs {x_label} (Half vs Quarter Filling)",
                x=0.5,
                xanchor='center',
                y=0.98,
                yanchor='top',
            ),
            hovermode='closest',
            height=750,
            width=400 * n_cols_this_page,
            margin=dict(b=80),
            **legend_configs,
        )
        fig.add_annotation(
            text=annotation_text,
            x=0.5,
            xref='paper',
            y=-0.12,
            yref='paper',
            showarrow=False,
            font=dict(size=11, color='gray'),
        )

        if save_html:
            page_suffix = f"_page_{page_idx + 1}" if n_pages > 1 else ""
            html_name = f"{filename_prefix}_L{L}_chi{chi}_{timestamp}{page_suffix}.html"
            html_path = os.path.join(output_dir, html_name)
            fig.write_html(html_path)
            html_paths.append(html_path)
            print(f"Saved figure to {html_path}")

        figures.append(fig)

        # --- FILLING PLOTS ---
        top_titles_fill = [f"Half-filling (n=1): {subplot_label}={sv}" for sv in current_subplot_values]
        bottom_titles_fill = [f"Quarter-filling (n=0.5): {subplot_label}={sv}" for sv in current_subplot_values]

        fig_fill = make_subplots(
            rows=2,
            cols=n_cols_this_page,
            subplot_titles=top_titles_fill + bottom_titles_fill,
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )

        legend_name_map_fill = {}
        legend_idx_fill = 1
        for r in range(1, 3):
            for c in range(1, n_cols_this_page + 1):
                if legend_idx_fill == 1:
                    legend_name_map_fill[(r, c)] = 'legend'
                else:
                    legend_name_map_fill[(r, c)] = f'legend{legend_idx_fill}'
                legend_idx_fill += 1

        any_trace_fill = False
        for col_idx, sp_val in enumerate(current_subplot_values):
            col = col_idx + 1
            sp_full_idx = subplot_list.index(sp_val)

            for row_idx, fill_mode in enumerate(filling_modes):
                row = row_idx + 1
                fill_idx = row_idx
                target_fill = filling_targets[fill_mode]

                # Add target filling reference line (horizontal)
                fig_fill.add_trace(
                    go.Scatter(
                        x=x_list_plot,
                        y=[target_fill] * len(x_list_plot),
                        mode='lines',
                        name=f"Target n={target_fill}",
                        legendgroup=f"target_{fill_mode}_{row}_{col}",
                        line=dict(color='#aaaaaa', width=1, dash='dash'),
                        showlegend=True,
                        hovertemplate=f"Target filling={target_fill}<extra></extra>",
                        legend=legend_name_map_fill[(row, col)],
                    ),
                    row=row,
                    col=col,
                )
                any_trace_fill = True

                # Add DMRG reference filling line
                dmrg_color = '#DAA520'
                if include_finite_dmrg:
                    dmrg_fill_xs, dmrg_fill_ys = [], []
                    for x_idx, x_val in enumerate(x_list_plot):
                        if x_param == 'U':
                            u_idx, v_idx = x_idx, sp_full_idx
                        else:
                            u_idx, v_idx = sp_full_idx, x_idx
                        dmrg_fill = finite_dmrg_fill_cache[u_idx, v_idx, fill_idx]
                        if np.isfinite(dmrg_fill):
                            dmrg_fill_xs.append(x_val)
                            dmrg_fill_ys.append(dmrg_fill)
                    if dmrg_fill_xs:
                        fig_fill.add_trace(
                            go.Scatter(
                                x=dmrg_fill_xs,
                                y=dmrg_fill_ys,
                                mode='lines+markers',
                                name="DMRG",
                                legendgroup=f"DMRG_fill_{row}_{col}",
                                marker=dict(color=dmrg_color, size=10, symbol='diamond'),
                                line=dict(color=dmrg_color, width=2, dash='dot'),
                                showlegend=True,
                                hovertemplate=f"DMRG<br>{x_label}=%{{x}}<br>n=%{{y:.4f}}<extra></extra>",
                                legend=legend_name_map_fill[(row, col)],
                            ),
                            row=row,
                            col=col,
                        )
                        any_trace_fill = True

                # Add cluster filling lines for each N_c value
                for nc_idx, Nc in enumerate(cluster_sizes):
                    xs, ys, hover_text = [], [], []
                    for x_idx, x_val in enumerate(x_list_plot):
                        if x_param == 'U':
                            u_idx = x_idx
                            V_key = sp_val
                        else:
                            u_idx = sp_full_idx
                            V_key = x_val
                        key = (Nc, V_key, fill_mode)
                        c_fill = cluster_fillings[key][u_idx]
                        if not np.isfinite(c_fill):
                            continue
                        U_val = u_list[u_idx]
                        xs.append(x_val)
                        ys.append(c_fill)
                        hover_text.append(f"U={U_val}<br>V={V_key}<br>N_c={Nc}<br>n={c_fill:.4f}")

                    if not xs:
                        continue

                    trace_color = nc_color_map[Nc]
                    fig_fill.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode='lines+markers',
                            name=f"N_c={Nc}",
                            legendgroup=f"Nc_{Nc}_fill_{row}_{col}",
                            marker=dict(color=trace_color, size=8),
                            line=dict(color=trace_color, width=2),
                            showlegend=True,
                            hovertemplate="%{text}<extra></extra>",
                            text=hover_text,
                            legend=legend_name_map_fill[(row, col)],
                        ),
                        row=row,
                        col=col,
                    )
                    any_trace_fill = True

                # Axis labels for filling plots
                fig_fill.update_xaxes(title_text=x_label, row=row, col=col)
                fill_range = [0, 2]
                if shared_yaxis is not None and fill_y_ranges[fill_idx] is not None:
                    fill_range = fill_y_ranges[fill_idx]
                fig_fill.update_yaxes(
                    title_text="Filling per site",
                    row=row,
                    col=col,
                    type='linear',
                    tickformat='.3f',
                    range=fill_range,
                )

        if not any_trace_fill:
            warnings.warn("No valid filling data points available to plot.")
        else:
            legend_configs_fill = {}
            for col_idx_leg in range(n_cols_this_page):
                for row_idx_leg in range(2):
                    row_leg = row_idx_leg + 1
                    col_leg = col_idx_leg + 1
                    legend_key = legend_name_map_fill[(row_leg, col_leg)]

                    x_end = col_idx_leg * (subplot_width + h_spacing) + subplot_width
                    y_start = 1.0 - row_idx_leg * (subplot_height + v_spacing) - subplot_height

                    legend_configs_fill[legend_key] = dict(
                        x=x_end - 0.01,
                        y=y_start + 0.02,
                        xanchor='right',
                        yanchor='bottom',
                        bgcolor='rgba(255, 255, 255, 0.85)',
                        bordercolor='rgba(0, 0, 0, 0.3)',
                        borderwidth=1,
                        font=dict(size=8),
                        itemsizing='constant',
                        tracegroupgap=0,
                        itemwidth=30,
                    )

            fig_fill.update_layout(
                title=dict(
                    text=f"Filling per Site vs {x_label} (Half vs Quarter Filling)",
                    x=0.5,
                    xanchor='center',
                    y=0.98,
                    yanchor='top',
                ),
                hovermode='closest',
                height=750,
                width=400 * n_cols_this_page,
                margin=dict(b=80),
                **legend_configs_fill,
            )
            fig_fill.add_annotation(
                text=annotation_text,
                x=0.5,
                xref='paper',
                y=-0.12,
                yref='paper',
                showarrow=False,
                font=dict(size=11, color='gray'),
            )

            if save_html:
                page_suffix = f"_page_{page_idx + 1}" if n_pages > 1 else ""
                html_name_fill = f"{filename_prefix}_fillings_L{L}_chi{chi}_{timestamp}{page_suffix}.html"
                html_path_fill = os.path.join(output_dir, html_name_fill)
                fig_fill.write_html(html_path_fill)
                filling_html_paths.append(html_path_fill)
                print(f"Saved filling figure to {html_path_fill}")

            filling_figures.append(fig_fill)

        # --- CORRELATION LENGTH PLOTS (if compute_localization is enabled) ---
        if compute_localization and idmrg_corr_length_cache is not None and include_idmrg:
            top_titles_xi = [f"Half-filling (n=1): {subplot_label}={sv}" for sv in current_subplot_values]
            bottom_titles_xi = [f"Quarter-filling (n=0.5): {subplot_label}={sv}" for sv in current_subplot_values]

            fig_xi = make_subplots(
                rows=2,
                cols=n_cols_this_page,
                subplot_titles=top_titles_xi + bottom_titles_xi,
                horizontal_spacing=0.08,
                vertical_spacing=0.12,
            )

            legend_name_map_xi = {}
            legend_idx_xi = 1
            for r in range(1, 3):
                for c in range(1, n_cols_this_page + 1):
                    if legend_idx_xi == 1:
                        legend_name_map_xi[(r, c)] = 'legend'
                    else:
                        legend_name_map_xi[(r, c)] = f'legend{legend_idx_xi}'
                    legend_idx_xi += 1

            any_trace_xi = False
            for col_idx, sp_val in enumerate(current_subplot_values):
                col = col_idx + 1
                sp_full_idx = subplot_list.index(sp_val)

                for row_idx, fill_mode in enumerate(filling_modes):
                    row = row_idx + 1
                    fill_idx = row_idx

                    # Add iDMRG correlation length
                    idmrg_xi_xs, idmrg_xi_ys = [], []
                    for x_idx, x_val in enumerate(x_list_plot):
                        if x_param == 'U':
                            u_idx, v_idx = x_idx, sp_full_idx
                        else:
                            u_idx, v_idx = sp_full_idx, x_idx
                        xi_val = idmrg_corr_length_cache[u_idx, v_idx, fill_idx]
                        if np.isfinite(xi_val):
                            idmrg_xi_xs.append(x_val)
                            idmrg_xi_ys.append(xi_val)

                    if idmrg_xi_xs:
                        dmrg_color = '#DAA520'
                        fig_xi.add_trace(
                            go.Scatter(
                                x=idmrg_xi_xs,
                                y=idmrg_xi_ys,
                                mode='lines+markers',
                                name="iDMRG",
                                legendgroup=f"iDMRG_xi_{row}_{col}",
                                marker=dict(color=dmrg_color, size=10, symbol='diamond'),
                                line=dict(color=dmrg_color, width=2),
                                showlegend=True,
                                hovertemplate=f"iDMRG<br>{x_label}=%{{x}}<br>ξ=%{{y:.4f}}<extra></extra>",
                                legend=legend_name_map_xi[(row, col)],
                            ),
                            row=row,
                            col=col,
                        )
                        any_trace_xi = True

                    # Axis labels for correlation length plots
                    fig_xi.update_xaxes(title_text=x_label, row=row, col=col)
                    fig_xi.update_yaxes(
                        title_text="Correlation length ξ",
                        row=row,
                        col=col,
                        type='linear',
                    )

            if any_trace_xi:
                legend_configs_xi = {}
                for col_idx_leg in range(n_cols_this_page):
                    for row_idx_leg in range(2):
                        row_leg = row_idx_leg + 1
                        col_leg = col_idx_leg + 1
                        legend_key = legend_name_map_xi[(row_leg, col_leg)]

                        x_end = col_idx_leg * (subplot_width + h_spacing) + subplot_width
                        y_start = 1.0 - row_idx_leg * (subplot_height + v_spacing) - subplot_height

                        legend_configs_xi[legend_key] = dict(
                            x=x_end - 0.01,
                            y=y_start + 0.02,
                            xanchor='right',
                            yanchor='bottom',
                            bgcolor='rgba(255, 255, 255, 0.85)',
                            bordercolor='rgba(0, 0, 0, 0.3)',
                            borderwidth=1,
                            font=dict(size=8),
                            itemsizing='constant',
                            tracegroupgap=0,
                            itemwidth=30,
                        )

                fig_xi.update_layout(
                    title=dict(
                        text=f"Correlation Length vs {x_label} (Half vs Quarter Filling)",
                        x=0.5,
                        xanchor='center',
                        y=0.98,
                        yanchor='top',
                    ),
                    hovermode='closest',
                    height=750,
                    width=400 * n_cols_this_page,
                    margin=dict(b=80),
                    **legend_configs_xi,
                )
                fig_xi.add_annotation(
                    text=annotation_text,
                    x=0.5,
                    xref='paper',
                    y=-0.12,
                    yref='paper',
                    showarrow=False,
                    font=dict(size=11, color='gray'),
                )

                if save_html:
                    page_suffix = f"_page_{page_idx + 1}" if n_pages > 1 else ""
                    html_name_xi = f"{filename_prefix}_corr_length_L{L}_chi{chi}_{timestamp}{page_suffix}.html"
                    html_path_xi = os.path.join(output_dir, html_name_xi)
                    fig_xi.write_html(html_path_xi)
                    corr_length_html_paths.append(html_path_xi)
                    print(f"Saved correlation length figure to {html_path_xi}")

                corr_length_figures.append(fig_xi)

    fig = figures[0] if figures else None
    saved_paths = {}
    if save_html:
        saved_paths['html_pages'] = html_paths
        saved_paths['filling_html_pages'] = filling_html_paths
        saved_paths['corr_length_html_pages'] = corr_length_html_paths
        if html_paths:
            saved_paths['html'] = html_paths[0]
        if filling_html_paths:
            saved_paths['filling_html'] = filling_html_paths[0]
        if corr_length_html_paths:
            saved_paths['corr_length_html'] = corr_length_html_paths[0]

    # Serialize results
    cluster_energy_serialized: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    cluster_fillings_serialized: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for Nc in cluster_sizes:
        cluster_energy_serialized[str(Nc)] = {}
        cluster_fillings_serialized[str(Nc)] = {}
        for V in v_list:
            v_key = str(V)
            cluster_energy_serialized[str(Nc)][v_key] = {}
            cluster_fillings_serialized[str(Nc)][v_key] = {}
            for fill_mode in filling_modes:
                key = (Nc, V, fill_mode)
                cluster_energy_serialized[str(Nc)][v_key][fill_mode] = cluster_results[key].tolist()
                cluster_fillings_serialized[str(Nc)][v_key][fill_mode] = cluster_fillings[key].tolist()

    int_sep_map_serialized = {str(Nc): list(ratio) for Nc, ratio in ratio_map.items()}

    results_payload = {
        'cluster_sizes': cluster_sizes,
        'int_sep_ratios': int_sep_map_serialized,
        'U_values': u_list,
        'V_values': v_list,
        'filling_modes': filling_modes,
        'cluster_energies': cluster_energy_serialized,
        'cluster_fillings': cluster_fillings_serialized,
        'idmrg_energies': idmrg_cache.tolist(),
        'finite_dmrg_energies': finite_dmrg_cache.tolist(),
        'idmrg_fillings': idmrg_fill_cache.tolist(),
        'finite_dmrg_fillings': finite_dmrg_fill_cache.tolist(),
        'idmrg_correlation_lengths': idmrg_corr_length_cache.tolist() if idmrg_corr_length_cache is not None else None,
        'parameters': {
            'v_sep_ratio': v_sep_ratio,
            't': t,
            'L': L,
            'chi': chi,
            'solver_method': solver_method,
            'states_retained': states_retained,
            'include_idmrg': include_idmrg,
            'include_finite_dmrg': include_finite_dmrg,
            'plot_relative_error': plot_relative_error,
            'dmrg_fixed_filling': dmrg_fixed_filling,
        },
        'artifacts': saved_paths,
        'failures': failed_calculations,
    }

    if include_timing and timing_recorder is not None:
        results_payload['timings'] = timing_recorder.records
        if include_timing_plot and timing_recorder.records:
            fig_timing, timing_artifacts = plot_timings(
                timing_recorder.records,
                output_dir=output_dir,
                filename_prefix=f"{filename_prefix}_timing",
                show_plots=show_plots,
            )
            saved_paths['timing_plot'] = timing_artifacts.get('html')

    if save_data:
        pickle_name = f"{filename_prefix}_L{L}_chi{chi}_{timestamp}.pkl"
        pickle_path = os.path.join(output_dir, pickle_name)
        with open(pickle_path, 'wb') as fh:
            pickle.dump(results_payload, fh)
        saved_paths['pickle'] = pickle_path
        print(f"Saved data to {pickle_path}")

    if failed_calculations:
        print("\n" + "=" * 60)
        print("WARNING: Some calculations failed")
        print("=" * 60)
        for failure in failed_calculations:
            params_desc = ', '.join(f"{k}={v}" for k, v in failure['params'].items())
            print(f"{failure['method']}: {params_desc}")
            error_lines = failure['error'].splitlines()
            print(f"  Error: {error_lines[0] if error_lines else '(no error message)'}")
            if 'traceback' in failure:
                print(f"  Full traceback:\n{failure['traceback']}")

    if show_plots:
        for energy_fig in figures:
            energy_fig.show()
        for fig_fill in filling_figures:
            fig_fill.show()
        for fig_xi in corr_length_figures:
            fig_xi.show()

    return fig, results_payload


def compute_supercluster_size(
    L: int,
    int_sep_ratio: Tuple[int, int],
    v_sep_ratio: Tuple[int, int],
) -> int:
    """
    Compute the supercluster size for given L, int_sep_ratio, and v_sep_ratio.

    The supercluster size is L / gcd(L, m, n) where:
    - m = L * int_sep_ratio[0] / int_sep_ratio[1] (interaction separation in lattice units)
    - n = L * v_sep_ratio[0] / v_sep_ratio[1] (V separation in lattice units)

    Args:
        L: System size
        int_sep_ratio: Interaction separation ratio (p, q) giving step = L*p/q
        v_sep_ratio: V separation ratio (p, q) giving step = L*p/q

    Returns:
        The supercluster size (number of sites in each supercluster)
    """
    from math import gcd

    # Compute lattice steps
    m = int(L * int_sep_ratio[0] / int_sep_ratio[1])
    n = int(L * v_sep_ratio[0] / v_sep_ratio[1])

    # gcd of three numbers
    g = gcd(gcd(L, m), n)

    return L // g


def compatible_int_seps(
    L: int,
    v_sep_ratio: Tuple[int, int],
    target_supercluster_size: int,
    Nc_values: Optional[Sequence[int]] = None,
) -> List[Tuple[int, int]]:
    """
    Find all int_sep_ratios that give a specific supercluster size for given L and v_sep_ratio.

    The supercluster size is L / gcd(L, m, n) where m = int_sep, n = v_sep.
    For a target size S, we need gcd(L, m, n) = L / S.

    Args:
        L: System size
        v_sep_ratio: V separation ratio (p, q)
        target_supercluster_size: Desired supercluster size
        Nc_values: If provided, only return int_seps compatible with these cluster sizes

    Returns:
        List of valid int_sep_ratios as (1, q) tuples where q divides L
    """
    from math import gcd

    n = int(L * v_sep_ratio[0] / v_sep_ratio[1])  # V lattice step
    target_gcd = L // target_supercluster_size

    if L % target_supercluster_size != 0:
        return []  # target_supercluster_size must divide L

    valid_ratios = []

    # Iterate over all divisors of L as potential denominators q
    # int_sep_ratio = (1, q) gives m = L/q
    for q in range(1, L + 1):
        if L % q != 0:
            continue
        m = L // q  # This is the lattice step for int_sep_ratio = (1, q)

        # Compute resulting supercluster size
        g = gcd(gcd(L, m), n)
        sc_size = L // g

        if sc_size == target_supercluster_size:
            int_sep_ratio = (1, q)

            # If Nc_values provided, check compatibility
            if Nc_values is not None:
                # Check that at least one Nc is compatible
                # Nc must divide qm = L / gcd(L, m)
                g1 = gcd(L, m)
                qm = L // g1
                compatible = any(qm % Nc == 0 for Nc in Nc_values)
                if not compatible:
                    continue

            valid_ratios.append(int_sep_ratio)

    return valid_ratios


def compare_fixed_supercluster(
    v_sep_ratio: Tuple[int, int],
    U_values: Sequence[float],
    V_values: Sequence[float],
    *,
    supercluster_int_seps: Optional[Dict[int, List[Tuple[int, Tuple[int, int]]]]] = None,
    supercluster_sizes: Optional[Sequence[int]] = None,
    max_seps: int = 3,
    set_filling: float = 1.0,
    t: float = 1.0,
    L: int = 20,
    chi: int = 32,
    solver_method: str = 'dense_ED',
    states_retained: int = 4,
    output_dir: str = 'large_files/plots',
    show_plots: bool = True,
    save_html: bool = True,
    save_data: bool = True,
    filename_prefix: str = 'fixed_supercluster_comparison',
    log_yaxis: bool = True,
    include_idmrg: bool = False,
    include_finite_dmrg: bool = True,
    include_timing: bool = False,
    include_timing_plot: bool = False,
    plot_relative_error: bool = False,
    dmrg_fixed_filling: bool = True,
    axes: Tuple[str, str] = ('U', 'V'),
    rows_per_page: int = 2,
    cols_per_page: int = 3,
    results: Optional[Union[Dict, str, os.PathLike]] = None,
) -> Tuple[go.Figure, Dict]:
    """
    Compare cluster calculations for fixed supercluster sizes.

    Creates a grid where:
    - Rows: U or V values (controlled by ``axes``)
    - Columns: Different supercluster sizes
    - X-axis: U or V values (controlled by ``axes``)
    - Lines: Different (Nc, int_sep) pairs that give the same supercluster size
           (maximal separation (1, Nc) shown in black)
    - Y-axis: Ground state energy per site (or relative error)

    Args:
        v_sep_ratio: Ratio controlling the AA modulation for V.
        U_values: U values.
        V_values: V values.
        supercluster_int_seps: Optional dict mapping supercluster_size -> list of (Nc, int_sep_ratio) pairs.
            Each entry is (Nc, (p, q)) where int_sep_ratio = (p, q).
            Each (Nc, int_sep) pair must produce the claimed supercluster size.
            The function will verify this and raise an error if mismatched.
            Example: {4: [(4, (1, 4)), (2, (1, 4))]} means SC size 4 with two options:
                - Nc=4, int_sep=(1,4)
                - Nc=2, int_sep=(1,4)
            If None, auto-generate using supercluster_sizes and max_seps.
        supercluster_sizes: List of supercluster sizes to compare. Required if supercluster_int_seps is None.
            Must be divisors of L.
        max_seps: Maximum number of (Nc, int_sep) pairs to generate per supercluster size
            when auto-generating. The maximal case (Nc=SC, int_sep=(1,SC)) is always included
            first if valid. Default is 3.
        set_filling: Target filling per site (default 1.0 for half-filling).
        t: Hopping parameter.
        L: System size.
        chi: Bond dimension for DMRG.
        solver_method: Diagonalization backend for cluster Hamiltonian.
        states_retained: Number of states retained in cluster solver.
        output_dir: Directory for saved artifacts.
        show_plots: Whether to display the generated figure.
        save_html: If True, save the interactive figure as HTML.
        save_data: If True, pickle the raw numerical results.
        filename_prefix: Prefix for saved artifact names.
        log_yaxis: Plot y-axis on log scale (for relative error plots).
        include_idmrg: Whether to include iDMRG reference.
        include_finite_dmrg: Whether to include finite DMRG reference.
        include_timing: Whether to record timing information.
        include_timing_plot: Whether to generate timing plots.
        plot_relative_error: If True, show relative error instead of energy.
        dmrg_fixed_filling: If True, DMRG uses canonical ensemble (fixed N).
        axes: Controls which parameter is plotted on the x-axis vs. rows.
            ('U', 'V') — U on x-axis, V indexes rows (default).
            ('V', 'U') — V on x-axis, U indexes rows.
            Supercluster sizes always index columns.
        rows_per_page: Number of rows per page.
        cols_per_page: Number of columns (supercluster sizes) per page.
        results: Pre-computed results dict or path to pickle file.

    Returns:
        (figure, results_dict)
    """
    if t is None:
        raise ValueError("Parameter t must be specified.")

    def _coerce_ratio(value, label: str) -> Tuple[int, int]:
        if value is None:
            raise ValueError(f"{label} ratio must be provided.")
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{label} ratio must be a length-2 iterable, got {value!r}.")
        try:
            p = int(round(value[0]))
            q = int(round(value[1]))
        except Exception as exc:
            raise ValueError(f"Could not parse {label} ratio {value!r} into integers.") from exc
        if q == 0:
            raise ValueError(f"Denominator for {label} ratio cannot be zero.")
        return (p, q)

    v_sep_ratio = _coerce_ratio(v_sep_ratio, "V separation")

    U_values = np.asarray(U_values, dtype=float)
    if U_values.ndim != 1 or U_values.size == 0:
        raise ValueError("U_values must be a 1-D array with at least one entry.")
    u_list = [float(u) for u in U_values]

    V_values = np.asarray(V_values, dtype=float)
    if V_values.ndim != 1 or V_values.size == 0:
        raise ValueError("V_values must be a 1-D array with at least one entry.")
    v_list = [float(v) for v in V_values]

    from math import gcd

    # Auto-generate supercluster_int_seps if not provided
    if supercluster_int_seps is None:
        if supercluster_sizes is None:
            raise ValueError(
                "Either supercluster_int_seps or supercluster_sizes must be provided."
            )

        supercluster_int_seps = {}
        for sc_size in supercluster_sizes:
            if L % sc_size != 0:
                raise ValueError(
                    f"supercluster_size={sc_size} must divide L={L}"
                )

            # Find all valid int_seps for this supercluster size
            valid_int_seps = compatible_int_seps(L, v_sep_ratio, sc_size)
            if not valid_int_seps:
                warnings.warn(f"No valid int_seps found for sc_size={sc_size}, skipping.")
                continue

            # Generate (Nc, int_sep) pairs
            pairs: List[Tuple[int, Tuple[int, int]]] = []

            # Helper to check if int_sep is trivial (0 or 2π multiples)
            # int_sep = (p, q) gives separation = 2π * p / q
            # Trivial cases: p/q is integer (0, 2π, 4π, ...) i.e. q divides p
            def is_trivial_int_sep(int_sep: Tuple[int, int]) -> bool:
                p, q = int_sep
                return p % q == 0  # p/q is integer means 0, 2π, 4π, etc.

            # Filter out trivial int_seps
            valid_int_seps = [s for s in valid_int_seps if not is_trivial_int_sep(s)]

            # Always try to include the maximal case first: Nc=sc_size, int_sep=(1, sc_size)
            maximal_int_sep = (1, sc_size)
            if maximal_int_sep in valid_int_seps:
                # Check if Nc=sc_size is compatible with this int_sep
                m = L // sc_size
                g1 = gcd(L, m)
                qm = L // g1
                if qm % sc_size == 0:
                    pairs.append((sc_size, maximal_int_sep))

            # Add other (Nc, int_sep) combinations up to max_seps
            for int_sep in valid_int_seps:
                if len(pairs) >= max_seps:
                    break

                # Compute qm for this int_sep to find compatible Nc values
                m = int(L * int_sep[0] / int_sep[1])
                g1 = gcd(L, m)
                qm = L // g1

                # Find all Nc that divide qm (compatible cluster sizes), excluding Nc=1
                compatible_Nc_values = [nc for nc in range(2, qm + 1) if qm % nc == 0]

                for Nc in sorted(compatible_Nc_values, reverse=True):  # Prefer larger Nc
                    if len(pairs) >= max_seps:
                        break
                    pair = (Nc, int_sep)
                    if pair not in pairs:
                        pairs.append(pair)

            if pairs:
                supercluster_int_seps[sc_size] = pairs
                print(f"Auto-generated for SC={sc_size}: {pairs}")

        if not supercluster_int_seps:
            raise ValueError("No valid supercluster configurations could be generated.")

    # Validate and organize supercluster_int_seps
    # Input format: {sc_size: [(Nc, int_sep_ratio), ...]}
    sc_sizes = sorted(supercluster_int_seps.keys())
    if not sc_sizes:
        raise ValueError("supercluster_int_seps must have at least one entry.")

    # Validate that each (Nc, int_sep) pair produces the claimed supercluster size
    # validated_map[sc_size] = [(Nc, int_sep_ratio), ...]
    validated_map: Dict[int, List[Tuple[int, Tuple[int, int]]]] = {}
    for sc_size, nc_int_sep_list in supercluster_int_seps.items():
        validated_map[sc_size] = []
        for entry in nc_int_sep_list:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(
                    f"Each entry in supercluster_int_seps[{sc_size}] must be (Nc, int_sep_ratio), got {entry!r}"
                )
            Nc = int(entry[0])
            int_sep = _coerce_ratio(entry[1], f"int_sep for sc_size={sc_size}, Nc={Nc}")

            # Validate supercluster size
            computed_sc_size = compute_supercluster_size(L, int_sep, v_sep_ratio)
            if computed_sc_size != sc_size:
                raise ValueError(
                    f"(Nc={Nc}, int_sep={int_sep}) with v_sep={v_sep_ratio} gives supercluster_size={computed_sc_size}, "
                    f"not the claimed {sc_size}"
                )

            # Validate that Nc is compatible with int_sep (qm % Nc == 0)
            m = int(L * int_sep[0] / int_sep[1])
            g1 = gcd(L, m)
            qm = L // g1
            if qm % Nc != 0:
                raise ValueError(
                    f"Nc={Nc} is not compatible with int_sep={int_sep}: qm={qm} is not divisible by Nc={Nc}"
                )

            validated_map[sc_size].append((Nc, int_sep))

    # Timing setup
    timing_recorder = TimingRecorder() if include_timing else None
    failed_calculations: List[Dict] = []

    n_U = len(u_list)
    n_V = len(v_list)

    # Validate and set up axes mapping
    x_param, row_param = axes
    if x_param not in ('U', 'V') or row_param not in ('U', 'V') or x_param == row_param:
        raise ValueError(f"axes must be ('U','V') or ('V','U'), got {axes!r}")

    if x_param == 'U':
        x_list = u_list
        row_list = v_list
        x_label = 'U'
        row_label = 'V'
    else:
        x_list = v_list
        row_list = u_list
        x_label = 'V'
        row_label = 'U'

    def _vu_indices(row_i: int, x_i: int) -> Tuple[int, int]:
        """Map (row_index, x_index) -> (v_idx, u_idx) for cache lookups."""
        if x_param == 'U':
            return (row_i, x_i)
        return (x_i, row_i)

    n_rows_param = len(row_list)

    # Load or compute results
    if results is not None:
        if isinstance(results, (str, os.PathLike)):
            with open(results, 'rb') as fh:
                results = pickle.load(fh)
        print("Using precomputed results payload; skipping new simulations.")

        # Extract cached data from results payload
        cluster_results: Dict[Tuple[int, int, Tuple[int, int]], np.ndarray] = {}
        cluster_fillings: Dict[Tuple[int, int, Tuple[int, int]], np.ndarray] = {}

        # Deserialize cluster results: {sc_key: {pair_key: [[values]]}} -> {(sc, Nc, int_sep): array}
        cluster_energy_serialized = results.get('cluster_energies', {})
        cluster_fillings_serialized = results.get('cluster_fillings', {})

        for sc_key, pairs_dict in cluster_energy_serialized.items():
            sc_size = int(sc_key)
            for pair_key, values in pairs_dict.items():
                # pair_key format: "Nc_p_q"
                parts = pair_key.split('_')
                Nc = int(parts[0])
                int_sep = (int(parts[1]), int(parts[2]))
                key = (sc_size, Nc, int_sep)
                cluster_results[key] = np.array(values, dtype=float)

        for sc_key, pairs_dict in cluster_fillings_serialized.items():
            sc_size = int(sc_key)
            for pair_key, values in pairs_dict.items():
                parts = pair_key.split('_')
                Nc = int(parts[0])
                int_sep = (int(parts[1]), int(parts[2]))
                key = (sc_size, Nc, int_sep)
                cluster_fillings[key] = np.array(values, dtype=float)

        # Deserialize DMRG caches
        idmrg_data = results.get('idmrg_energies', [])
        finite_dmrg_data = results.get('finite_dmrg_energies', [])
        idmrg_fill_data = results.get('idmrg_fillings', [])
        finite_dmrg_fill_data = results.get('finite_dmrg_fillings', [])

        if idmrg_data:
            idmrg_cache = np.array(idmrg_data, dtype=float)
        else:
            idmrg_cache = np.full((n_V, n_U), np.nan, dtype=float)

        if finite_dmrg_data:
            finite_dmrg_cache = np.array(finite_dmrg_data, dtype=float)
        else:
            finite_dmrg_cache = np.full((n_V, n_U), np.nan, dtype=float)

        if idmrg_fill_data:
            idmrg_fill_cache = np.array(idmrg_fill_data, dtype=float)
        else:
            idmrg_fill_cache = np.full((n_V, n_U), np.nan, dtype=float)

        if finite_dmrg_fill_data:
            finite_dmrg_fill_cache = np.array(finite_dmrg_fill_data, dtype=float)
        else:
            finite_dmrg_fill_cache = np.full((n_V, n_U), np.nan, dtype=float)

        # Don't re-save cached data
        save_data = False
        failed_calculations: List[Dict] = results.get('failures', [])
    else:
        # Initialize result caches
        # cluster_results[(sc_size, Nc, int_sep)][v_idx, u_idx] = energy_per_site
        cluster_results: Dict[Tuple[int, int, Tuple[int, int]], np.ndarray] = {}
        cluster_fillings: Dict[Tuple[int, int, Tuple[int, int]], np.ndarray] = {}

        for sc_size in sc_sizes:
            for Nc, int_sep in validated_map[sc_size]:
                key = (sc_size, Nc, int_sep)
                cluster_results[key] = np.full((n_V, n_U), np.nan, dtype=float)
                cluster_fillings[key] = np.full((n_V, n_U), np.nan, dtype=float)

        # DMRG cache: shape (n_V, n_U)
        idmrg_cache = np.full((n_V, n_U), np.nan, dtype=float)
        finite_dmrg_cache = np.full((n_V, n_U), np.nan, dtype=float)
        idmrg_fill_cache = np.full((n_V, n_U), np.nan, dtype=float)
        finite_dmrg_fill_cache = np.full((n_V, n_U), np.nan, dtype=float)

        # Compute DMRG references for each (V, U) pair
        print("=" * 60)
        print(f"Computing DMRG reference energies (filling={set_filling})")
        print("=" * 60)

        total_dmrg = n_V * n_U
        with tqdm(total=total_dmrg, desc="DMRG", ncols=80) as pbar:
            for v_idx, V in enumerate(v_list):
                for u_idx, U in enumerate(u_list):
                    # iDMRG
                    if include_idmrg:
                        try:
                            mu_0 = U / 2.0 if set_filling >= 0.9 else 0.0
                            energy_dmrg, filling_dmrg, _ = run_dmrg_method(U, mu_0, V, v_sep_ratio, t, L, chi)
                            dmrg_value = energy_dmrg + mu_0 * filling_dmrg
                            idmrg_cache[v_idx, u_idx] = dmrg_value
                            idmrg_fill_cache[v_idx, u_idx] = filling_dmrg
                        except Exception as exc:
                            import traceback
                            error_msg = str(exc) or f"{type(exc).__name__}: {repr(exc)}"
                            failed_calculations.append({
                                'method': 'iDMRG',
                                'params': {'U': U, 'V': V},
                                'error': error_msg,
                                'traceback': traceback.format_exc(),
                            })

                    # Finite DMRG
                    if include_finite_dmrg:
                        try:
                            if dmrg_fixed_filling:
                                energy_finite, _, filling_finite = get_gnd_fixed_filling(
                                    L, chi, set_filling, U, t, V, v_sep_ratio
                                )
                            else:
                                mu_0 = U / 2.0 if set_filling >= 0.9 else 0.0
                                energy_finite, _, filling_finite = get_gnd(
                                    L, chi, U, t, mu_0, V, v_sep_ratio
                                )
                            energy_finite_per_site = energy_finite / L
                            finite_dmrg_cache[v_idx, u_idx] = energy_finite_per_site
                            finite_dmrg_fill_cache[v_idx, u_idx] = filling_finite
                        except Exception as exc:
                            import traceback
                            error_msg = str(exc) or f"{type(exc).__name__}: {repr(exc)}"
                            failed_calculations.append({
                                'method': 'Finite DMRG',
                                'params': {'U': U, 'V': V},
                                'error': error_msg,
                                'traceback': traceback.format_exc(),
                            })
                    pbar.update(1)

        # Compute cluster results
        print("\n" + "=" * 60)
        print("Running cluster calculations")
        print("=" * 60)

        # Count total calculations: for each sc_size, count (Nc, int_sep) pairs * n_V * n_U
        total_calcs = sum(len(validated_map[sc]) for sc in sc_sizes) * n_V * n_U
        with tqdm(total=total_calcs, desc="Cluster calcs", ncols=80) as pbar:
            for sc_size in sc_sizes:
                for Nc, int_sep in validated_map[sc_size]:
                    # (Nc, int_sep) compatibility already validated during input parsing
                    for v_idx, V in enumerate(v_list):
                        for u_idx, U in enumerate(u_list):
                            try:
                                physical_params = PhysicalParams(U=U, mu_0=0.0, V=V, t=t)
                                run_config = ClusterModelConfig(
                                    L=L,
                                    int_cluster_size=Nc,
                                    cluster_separation_ratio=int_sep,
                                    V_separation_ratio=v_sep_ratio,
                                    ham_lib='quspin',
                                    physical_params=physical_params,
                                    model_bc='periodic',
                                    int_cluster_bc='periodic',
                                    super_cluster_bc='periodic',
                                    solver_method=solver_method,
                                    states_retained=states_retained,
                                )
                                system_expectations, _ = get_general_expectations(
                                    run_config,
                                    timing_recorder=timing_recorder,
                                    set_filling=set_filling,
                                )
                                energy, filling, mu_eff = system_expectations
                                energy_per_site = energy / L

                                key = (sc_size, Nc, int_sep)
                                cluster_results[key][v_idx, u_idx] = energy_per_site
                                cluster_fillings[key][v_idx, u_idx] = filling / L
                            except Exception as exc:
                                import traceback
                                error_msg = str(exc) or f"{type(exc).__name__}: {repr(exc)}"
                                failed_calculations.append({
                                    'method': f'Cluster (sc={sc_size}, Nc={Nc}, int_sep={int_sep})',
                                    'params': {'U': U, 'V': V},
                                    'error': error_msg,
                                    'traceback': traceback.format_exc(),
                                })
                            pbar.update(1)

    # --- PLOTTING LOGIC ---
    # Color palette for different (Nc, int_sep) pairs
    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf',
    ]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)

    # Grid: rows = row_param values, cols = supercluster sizes
    n_rows_total = n_rows_param
    n_cols_total = len(sc_sizes)

    # Pagination
    n_row_pages = int(np.ceil(n_rows_total / rows_per_page))
    n_col_pages = int(np.ceil(n_cols_total / cols_per_page))
    n_pages = n_row_pages * n_col_pages

    figures: List[go.Figure] = []
    html_paths: List[str] = []

    page_idx = 0
    for row_page in range(n_row_pages):
        row_start = row_page * rows_per_page
        row_end = min(row_start + rows_per_page, n_rows_total)
        current_row_indices = list(range(row_start, row_end))
        n_rows_this_page = len(current_row_indices)

        for col_page in range(n_col_pages):
            col_start = col_page * cols_per_page
            col_end = min(col_start + cols_per_page, n_cols_total)
            current_sc_sizes = sc_sizes[col_start:col_end]
            n_cols_this_page = len(current_sc_sizes)

            # Subplot titles
            subplot_titles = []
            for ri in current_row_indices:
                row_val = row_list[ri]
                for sc_size in current_sc_sizes:
                    subplot_titles.append(f"{row_label}={row_val}, SC={sc_size}")

            fig = make_subplots(
                rows=n_rows_this_page,
                cols=n_cols_this_page,
                subplot_titles=subplot_titles,
                horizontal_spacing=0.08,
                vertical_spacing=0.10,
            )

            # Build legend name mapping
            legend_name_map = {}
            legend_idx = 1
            for r in range(1, n_rows_this_page + 1):
                for c in range(1, n_cols_this_page + 1):
                    if legend_idx == 1:
                        legend_name_map[(r, c)] = 'legend'
                    else:
                        legend_name_map[(r, c)] = f'legend{legend_idx}'
                    legend_idx += 1

            # First pass: collect all y-values for shared y-axis range
            all_y_values: List[float] = []
            for row_idx, ri in enumerate(current_row_indices):
                for col_idx, sc_size in enumerate(current_sc_sizes):
                    nc_int_sep_list = validated_map[sc_size]
                    # DMRG values
                    if include_finite_dmrg and not plot_relative_error:
                        for xi in range(len(x_list)):
                            v_idx, u_idx = _vu_indices(ri, xi)
                            if np.isfinite(finite_dmrg_cache[v_idx, u_idx]):
                                all_y_values.append(finite_dmrg_cache[v_idx, u_idx])
                    # Cluster values
                    for Nc, int_sep in nc_int_sep_list:
                        key = (sc_size, Nc, int_sep)
                        if key not in cluster_results:
                            continue
                        for xi in range(len(x_list)):
                            v_idx, u_idx = _vu_indices(ri, xi)
                            c_en = cluster_results[key][v_idx, u_idx]
                            if not np.isfinite(c_en):
                                continue
                            if plot_relative_error:
                                ref = finite_dmrg_cache[v_idx, u_idx]
                                if np.isfinite(ref) and ref != 0:
                                    all_y_values.append(np.abs(c_en - ref) / np.abs(ref))
                            else:
                                all_y_values.append(c_en)

            # Compute shared y-axis range with 5% padding
            if all_y_values:
                y_min, y_max = min(all_y_values), max(all_y_values)
                y_padding = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
                shared_y_range = [y_min - y_padding, y_max + y_padding]
            else:
                shared_y_range = None

            any_trace = False
            for row_idx, ri in enumerate(current_row_indices):
                row = row_idx + 1
                row_val = row_list[ri]

                for col_idx, sc_size in enumerate(current_sc_sizes):
                    col = col_idx + 1
                    nc_int_sep_list = validated_map[sc_size]  # List of (Nc, int_sep) pairs

                    # Create color map for (Nc, int_sep) pairs
                    # Only the TRUE maximal case is black: Nc = sc_size and int_sep = (1, sc_size)
                    # This is the case where cluster size equals supercluster size
                    pair_color_map = {}
                    non_maximal_pairs = []
                    true_maximal = (sc_size, (1, sc_size))  # (Nc, int_sep) for true maximal
                    for Nc, int_sep in nc_int_sep_list:
                        if (Nc, int_sep) == true_maximal:
                            pair_color_map[(Nc, int_sep)] = '#000000'  # Black for true maximal
                        else:
                            non_maximal_pairs.append((Nc, int_sep))

                    for idx, pair in enumerate(non_maximal_pairs):
                        pair_color_map[pair] = color_palette[idx % len(color_palette)]

                    # Add DMRG reference line (gold, dotted)
                    dmrg_color = '#DAA520'
                    if include_finite_dmrg and not plot_relative_error:
                        dmrg_xs, dmrg_ys = [], []
                        for xi, x_val in enumerate(x_list):
                            v_idx, u_idx = _vu_indices(ri, xi)
                            if np.isfinite(finite_dmrg_cache[v_idx, u_idx]):
                                dmrg_xs.append(x_val)
                                dmrg_ys.append(finite_dmrg_cache[v_idx, u_idx])
                        if dmrg_xs:
                            fig.add_trace(
                                go.Scatter(
                                    x=dmrg_xs,
                                    y=dmrg_ys,
                                    mode='lines+markers',
                                    name="DMRG",
                                    legendgroup=f"DMRG_{row}_{col}",
                                    marker=dict(color=dmrg_color, size=10, symbol='diamond'),
                                    line=dict(color=dmrg_color, width=2, dash='dot'),
                                    showlegend=True,
                                    hovertemplate=f"DMRG<br>{x_label}=%{{x}}<br>E=%{{y:.6f}}<extra></extra>",
                                    legend=legend_name_map[(row, col)],
                                ),
                                row=row,
                                col=col,
                            )
                            any_trace = True

                    # Add cluster lines for each (Nc, int_sep) pair
                    for Nc, int_sep in nc_int_sep_list:
                        key = (sc_size, Nc, int_sep)
                        if key not in cluster_results:
                            continue

                        xs, ys, hover_text = [], [], []
                        for xi, x_val in enumerate(x_list):
                            v_idx, u_idx = _vu_indices(ri, xi)
                            c_en = cluster_results[key][v_idx, u_idx]
                            if not np.isfinite(c_en):
                                continue

                            if plot_relative_error:
                                ref = finite_dmrg_cache[v_idx, u_idx]
                                if not (np.isfinite(ref) and ref != 0):
                                    continue
                                c_err = np.abs(c_en - ref) / np.abs(ref)
                                xs.append(x_val)
                                ys.append(c_err)
                                hover_text.append(
                                    f"{x_label}={x_val}<br>Nc={Nc}, int_sep={int_sep}<br>rel_err={c_err:.2%}"
                                )
                            else:
                                xs.append(x_val)
                                ys.append(c_en)
                                hover_text.append(
                                    f"{x_label}={x_val}<br>Nc={Nc}, int_sep={int_sep}<br>E={c_en:.6f}"
                                )

                        if not xs:
                            continue

                        trace_color = pair_color_map.get((Nc, int_sep), color_palette[0])
                        int_sep_label = format_sep_as_pi(int_sep)
                        # True maximal: Nc = sc_size and int_sep = (1, sc_size)
                        is_true_maximal = (Nc == sc_size and int_sep == (1, sc_size))
                        fig.add_trace(
                            go.Scatter(
                                x=xs,
                                y=ys,
                                mode='lines+markers',
                                name=f"Nc={Nc}, m={int_sep_label}" + (" (max)" if is_true_maximal else ""),
                                legendgroup=f"pair_{Nc}_{int_sep}_{row}_{col}",
                                marker=dict(color=trace_color, size=8),
                                line=dict(color=trace_color, width=2),
                                showlegend=True,
                                hovertemplate="%{text}<extra></extra>",
                                text=hover_text,
                                legend=legend_name_map[(row, col)],
                            ),
                            row=row,
                            col=col,
                        )
                        any_trace = True

                    # Axis labels with shared y-axis range
                    fig.update_xaxes(title_text=x_label, row=row, col=col)
                    y_title = "Relative error" if plot_relative_error else "Energy per site"
                    y_axis_kwargs = dict(
                        title_text=y_title,
                        type='log' if (plot_relative_error and log_yaxis) else 'linear',
                    )
                    if shared_y_range is not None and not (plot_relative_error and log_yaxis):
                        y_axis_kwargs['range'] = shared_y_range
                    fig.update_yaxes(row=row, col=col, **y_axis_kwargs)

            if not any_trace:
                warnings.warn(f"No valid data for page {page_idx + 1}")
                page_idx += 1
                continue

            # Layout and annotations
            v_sep_label = format_sep_as_pi(v_sep_ratio)
            annotation_parts = [
                f"v_sep={v_sep_label}",
                f"t={t}",
                f"L={L}",
                f"chi={chi}",
                f"filling={set_filling}",
            ]
            annotation_text = ", ".join(annotation_parts)
            if n_pages > 1:
                annotation_text += f" | Page {page_idx + 1}/{n_pages}"

            # Legend positioning
            h_spacing = 0.08
            v_spacing = 0.10
            subplot_width = (1.0 - h_spacing * (n_cols_this_page - 1)) / n_cols_this_page
            subplot_height = (1.0 - v_spacing * (n_rows_this_page - 1)) / n_rows_this_page

            legend_configs = {}
            for r_idx in range(n_rows_this_page):
                for c_idx in range(n_cols_this_page):
                    r = r_idx + 1
                    c = c_idx + 1
                    legend_key = legend_name_map[(r, c)]
                    x_end = c_idx * (subplot_width + h_spacing) + subplot_width
                    y_start = 1.0 - r_idx * (subplot_height + v_spacing) - subplot_height
                    legend_configs[legend_key] = dict(
                        x=x_end - 0.01,
                        y=y_start + 0.02,
                        xanchor='right',
                        yanchor='bottom',
                        bgcolor='rgba(255, 255, 255, 0.85)',
                        bordercolor='rgba(0, 0, 0, 0.3)',
                        borderwidth=1,
                        font=dict(size=8),
                        itemsizing='constant',
                        tracegroupgap=0,
                        itemwidth=30,
                    )

            fig.update_layout(
                title=dict(
                    text=f"Relative Error vs {x_label} (Fixed Supercluster Size)" if plot_relative_error
                         else f"Ground State Energy vs {x_label} (Fixed Supercluster Size)",
                    x=0.5,
                    xanchor='center',
                    y=0.98,
                    yanchor='top',
                ),
                hovermode='closest',
                height=350 * n_rows_this_page,
                width=400 * n_cols_this_page,
                margin=dict(b=80),
                **legend_configs,
            )
            fig.add_annotation(
                text=annotation_text,
                x=0.5,
                xref='paper',
                y=-0.08,
                yref='paper',
                showarrow=False,
                font=dict(size=11, color='gray'),
            )

            if save_html:
                page_suffix = f"_page_{page_idx + 1}" if n_pages > 1 else ""
                html_name = f"{filename_prefix}_L{L}_chi{chi}_{timestamp}{page_suffix}.html"
                html_path = os.path.join(output_dir, html_name)
                fig.write_html(html_path)
                html_paths.append(html_path)
                print(f"Saved figure to {html_path}")

            figures.append(fig)
            page_idx += 1

    # --- FILLING PLOTS ---
    # Same layout as energy plots: rows = row_param values, cols = SC sizes
    filling_figures: List[go.Figure] = []
    filling_html_paths: List[str] = []

    page_idx = 0
    for row_page in range(n_row_pages):
        row_start = row_page * rows_per_page
        row_end = min(row_start + rows_per_page, n_rows_total)
        current_row_indices = list(range(row_start, row_end))
        n_rows_this_page = len(current_row_indices)

        for col_page in range(n_col_pages):
            col_start = col_page * cols_per_page
            col_end = min(col_start + cols_per_page, n_cols_total)
            current_sc_sizes = sc_sizes[col_start:col_end]
            n_cols_this_page = len(current_sc_sizes)

            # Subplot titles
            subplot_titles_fill = []
            for ri in current_row_indices:
                row_val = row_list[ri]
                for sc_size in current_sc_sizes:
                    subplot_titles_fill.append(f"{row_label}={row_val}, SC={sc_size}")

            fig_fill = make_subplots(
                rows=n_rows_this_page,
                cols=n_cols_this_page,
                subplot_titles=subplot_titles_fill,
                horizontal_spacing=0.08,
                vertical_spacing=0.10,
            )

            # Build legend name mapping for filling figure
            legend_name_map_fill = {}
            legend_idx_fill = 1
            for r in range(1, n_rows_this_page + 1):
                for c in range(1, n_cols_this_page + 1):
                    if legend_idx_fill == 1:
                        legend_name_map_fill[(r, c)] = 'legend'
                    else:
                        legend_name_map_fill[(r, c)] = f'legend{legend_idx_fill}'
                    legend_idx_fill += 1

            # First pass: collect all filling y-values for shared y-axis range
            all_fill_values: List[float] = [set_filling]  # Include target filling
            for row_idx, ri in enumerate(current_row_indices):
                for col_idx, sc_size in enumerate(current_sc_sizes):
                    nc_int_sep_list = validated_map[sc_size]
                    # DMRG filling values
                    if include_finite_dmrg:
                        for xi in range(len(x_list)):
                            v_idx, u_idx = _vu_indices(ri, xi)
                            if np.isfinite(finite_dmrg_fill_cache[v_idx, u_idx]):
                                all_fill_values.append(finite_dmrg_fill_cache[v_idx, u_idx])
                    # Cluster filling values
                    for Nc, int_sep in nc_int_sep_list:
                        key = (sc_size, Nc, int_sep)
                        if key not in cluster_fillings:
                            continue
                        for xi in range(len(x_list)):
                            v_idx, u_idx = _vu_indices(ri, xi)
                            c_fill = cluster_fillings[key][v_idx, u_idx]
                            if np.isfinite(c_fill):
                                all_fill_values.append(c_fill)

            # Compute shared y-axis range for filling with 5% padding
            if all_fill_values:
                fill_y_min, fill_y_max = min(all_fill_values), max(all_fill_values)
                fill_y_padding = (fill_y_max - fill_y_min) * 0.05 if fill_y_max != fill_y_min else 0.1
                shared_fill_y_range = [fill_y_min - fill_y_padding, fill_y_max + fill_y_padding]
            else:
                shared_fill_y_range = None

            any_trace_fill = False
            for row_idx, ri in enumerate(current_row_indices):
                row = row_idx + 1
                row_val = row_list[ri]

                for col_idx, sc_size in enumerate(current_sc_sizes):
                    col = col_idx + 1
                    nc_int_sep_list = validated_map[sc_size]

                    # Create color map (same as energy plots)
                    pair_color_map_fill = {}
                    non_maximal_pairs_fill = []
                    true_maximal = (sc_size, (1, sc_size))
                    for Nc, int_sep in nc_int_sep_list:
                        if (Nc, int_sep) == true_maximal:
                            pair_color_map_fill[(Nc, int_sep)] = '#000000'
                        else:
                            non_maximal_pairs_fill.append((Nc, int_sep))
                    for idx, pair in enumerate(non_maximal_pairs_fill):
                        pair_color_map_fill[pair] = color_palette[idx % len(color_palette)]

                    # Add target filling reference line
                    fig_fill.add_trace(
                        go.Scatter(
                            x=list(x_list),
                            y=[set_filling] * len(x_list),
                            mode='lines',
                            name=f"Target n={set_filling}",
                            legendgroup=f"target_{row}_{col}",
                            line=dict(color='#aaaaaa', width=1, dash='dash'),
                            showlegend=True,
                            hovertemplate=f"Target filling={set_filling}<extra></extra>",
                            legend=legend_name_map_fill[(row, col)],
                        ),
                        row=row,
                        col=col,
                    )
                    any_trace_fill = True

                    # Add DMRG reference filling line (gold, dotted)
                    dmrg_color = '#DAA520'
                    if include_finite_dmrg:
                        dmrg_fill_xs, dmrg_fill_ys = [], []
                        for xi, x_val in enumerate(x_list):
                            v_idx, u_idx = _vu_indices(ri, xi)
                            if np.isfinite(finite_dmrg_fill_cache[v_idx, u_idx]):
                                dmrg_fill_xs.append(x_val)
                                dmrg_fill_ys.append(finite_dmrg_fill_cache[v_idx, u_idx])
                        if dmrg_fill_xs:
                            fig_fill.add_trace(
                                go.Scatter(
                                    x=dmrg_fill_xs,
                                    y=dmrg_fill_ys,
                                    mode='lines+markers',
                                    name="DMRG",
                                    legendgroup=f"DMRG_fill_{row}_{col}",
                                    marker=dict(color=dmrg_color, size=10, symbol='diamond'),
                                    line=dict(color=dmrg_color, width=2, dash='dot'),
                                    showlegend=True,
                                    hovertemplate=f"{x_label}=%{{x:.3g}}<br>n_DMRG=%{{y:.4f}}<extra></extra>",
                                    legend=legend_name_map_fill[(row, col)],
                                ),
                                row=row,
                                col=col,
                            )
                            any_trace_fill = True

                    # Add cluster filling lines for each (Nc, int_sep) pair
                    for Nc, int_sep in nc_int_sep_list:
                        key = (sc_size, Nc, int_sep)
                        if key not in cluster_fillings:
                            continue

                        xs, ys, hover_text = [], [], []
                        for xi, x_val in enumerate(x_list):
                            v_idx, u_idx = _vu_indices(ri, xi)
                            c_fill = cluster_fillings[key][v_idx, u_idx]
                            if not np.isfinite(c_fill):
                                continue
                            xs.append(x_val)
                            ys.append(c_fill)
                            hover_text.append(
                                f"{x_label}={x_val:.3g}<br>Nc={Nc}, int_sep={int_sep}<br>n={c_fill:.4f}"
                            )

                        if not xs:
                            continue

                        trace_color = pair_color_map_fill.get((Nc, int_sep), color_palette[0])
                        int_sep_label = format_sep_as_pi(int_sep)
                        is_true_maximal = (Nc == sc_size and int_sep == (1, sc_size))
                        fig_fill.add_trace(
                            go.Scatter(
                                x=xs,
                                y=ys,
                                mode='lines+markers',
                                name=f"Nc={Nc}, m={int_sep_label}" + (" (max)" if is_true_maximal else ""),
                                legendgroup=f"pair_fill_{Nc}_{int_sep}_{row}_{col}",
                                marker=dict(color=trace_color, size=8),
                                line=dict(color=trace_color, width=2),
                                showlegend=True,
                                hovertemplate="%{text}<extra></extra>",
                                text=hover_text,
                                legend=legend_name_map_fill[(row, col)],
                            ),
                            row=row,
                            col=col,
                        )
                        any_trace_fill = True

                    # Axis labels for filling plots with shared y-axis range
                    fig_fill.update_xaxes(title_text=x_label, row=row, col=col)
                    fill_y_axis_kwargs = dict(
                        title_text="Filling per site",
                        type='linear',
                        tickformat='.3f',
                    )
                    if shared_fill_y_range is not None:
                        fill_y_axis_kwargs['range'] = shared_fill_y_range
                    fig_fill.update_yaxes(row=row, col=col, **fill_y_axis_kwargs)

            if not any_trace_fill:
                warnings.warn(f"No valid filling data for page {page_idx + 1}")
                page_idx += 1
                continue

            # Layout and annotations for filling plot
            v_sep_label = format_sep_as_pi(v_sep_ratio)
            annotation_parts_fill = [
                f"v_sep={v_sep_label}",
                f"t={t}",
                f"L={L}",
                f"chi={chi}",
                f"target_filling={set_filling}",
            ]
            annotation_text_fill = ", ".join(annotation_parts_fill)
            if n_pages > 1:
                annotation_text_fill += f" | Page {page_idx + 1}/{n_pages}"

            # Legend positioning
            h_spacing = 0.08
            v_spacing = 0.10
            subplot_width = (1.0 - h_spacing * (n_cols_this_page - 1)) / n_cols_this_page
            subplot_height = (1.0 - v_spacing * (n_rows_this_page - 1)) / n_rows_this_page

            legend_configs_fill = {}
            for r_idx in range(n_rows_this_page):
                for c_idx in range(n_cols_this_page):
                    r = r_idx + 1
                    c = c_idx + 1
                    legend_key = legend_name_map_fill[(r, c)]
                    x_end = c_idx * (subplot_width + h_spacing) + subplot_width
                    y_start = 1.0 - r_idx * (subplot_height + v_spacing) - subplot_height
                    legend_configs_fill[legend_key] = dict(
                        x=x_end - 0.01,
                        y=y_start + 0.02,
                        xanchor='right',
                        yanchor='bottom',
                        bgcolor='rgba(255, 255, 255, 0.85)',
                        bordercolor='rgba(0, 0, 0, 0.3)',
                        borderwidth=1,
                        font=dict(size=8),
                        itemsizing='constant',
                        tracegroupgap=0,
                        itemwidth=30,
                    )

            fig_fill.update_layout(
                title=dict(
                    text=f"Filling per Site vs {x_label} (Fixed Supercluster Size)",
                    x=0.5,
                    xanchor='center',
                    y=0.98,
                    yanchor='top',
                ),
                hovermode='closest',
                height=350 * n_rows_this_page,
                width=400 * n_cols_this_page,
                margin=dict(b=80),
                **legend_configs_fill,
            )
            fig_fill.add_annotation(
                text=annotation_text_fill,
                x=0.5,
                xref='paper',
                y=-0.08,
                yref='paper',
                showarrow=False,
                font=dict(size=11, color='gray'),
            )

            if save_html:
                page_suffix = f"_page_{page_idx + 1}" if n_pages > 1 else ""
                html_name_fill = f"{filename_prefix}_fillings_L{L}_chi{chi}_{timestamp}{page_suffix}.html"
                html_path_fill = os.path.join(output_dir, html_name_fill)
                fig_fill.write_html(html_path_fill)
                filling_html_paths.append(html_path_fill)
                print(f"Saved filling figure to {html_path_fill}")

            filling_figures.append(fig_fill)
            page_idx += 1

    fig = figures[0] if figures else None
    saved_paths = {}
    if save_html:
        saved_paths['html_pages'] = html_paths
        if html_paths:
            saved_paths['html'] = html_paths[0]
        saved_paths['filling_html_pages'] = filling_html_paths
        if filling_html_paths:
            saved_paths['filling_html'] = filling_html_paths[0]

    # Serialize results
    # Format: cluster_energies[sc_size][f"{Nc}_{int_sep[0]}_{int_sep[1]}"] = 2D list (V x U)
    cluster_energy_serialized: Dict[str, Dict[str, List[List[float]]]] = {}
    cluster_fillings_serialized: Dict[str, Dict[str, List[List[float]]]] = {}
    for sc_size in sc_sizes:
        sc_key = str(sc_size)
        cluster_energy_serialized[sc_key] = {}
        cluster_fillings_serialized[sc_key] = {}
        for Nc, int_sep in validated_map[sc_size]:
            pair_key = f"{Nc}_{int_sep[0]}_{int_sep[1]}"
            key = (sc_size, Nc, int_sep)
            if key in cluster_results:
                cluster_energy_serialized[sc_key][pair_key] = cluster_results[key].tolist()
                cluster_fillings_serialized[sc_key][pair_key] = cluster_fillings[key].tolist()

    # Serialize validated_map: {sc_size: [(Nc, int_sep), ...]} -> {str: [[Nc, [p, q]], ...]}
    validated_map_serialized = {
        str(k): [[Nc, list(int_sep)] for Nc, int_sep in v]
        for k, v in validated_map.items()
    }

    results_payload = {
        'supercluster_sizes': sc_sizes,
        'supercluster_int_seps': validated_map_serialized,
        'U_values': u_list,
        'V_values': v_list,
        'set_filling': set_filling,
        'cluster_energies': cluster_energy_serialized,
        'cluster_fillings': cluster_fillings_serialized,
        'idmrg_energies': idmrg_cache.tolist() if 'idmrg_cache' in dir() else [],
        'finite_dmrg_energies': finite_dmrg_cache.tolist() if 'finite_dmrg_cache' in dir() else [],
        'idmrg_fillings': idmrg_fill_cache.tolist() if 'idmrg_fill_cache' in dir() else [],
        'finite_dmrg_fillings': finite_dmrg_fill_cache.tolist() if 'finite_dmrg_fill_cache' in dir() else [],
        'parameters': {
            'v_sep_ratio': v_sep_ratio,
            't': t,
            'L': L,
            'chi': chi,
            'solver_method': solver_method,
            'states_retained': states_retained,
            'include_idmrg': include_idmrg,
            'include_finite_dmrg': include_finite_dmrg,
            'plot_relative_error': plot_relative_error,
            'dmrg_fixed_filling': dmrg_fixed_filling,
        },
        'artifacts': saved_paths,
        'failures': failed_calculations,
    }

    if save_data:
        pickle_name = f"{filename_prefix}_L{L}_chi{chi}_{timestamp}.pkl"
        pickle_path = os.path.join(output_dir, pickle_name)
        with open(pickle_path, 'wb') as fh:
            pickle.dump(results_payload, fh)
        saved_paths['pickle'] = pickle_path
        print(f"Saved data to {pickle_path}")

    if failed_calculations:
        print("\n" + "=" * 60)
        print("WARNING: Some calculations failed")
        print("=" * 60)
        for failure in failed_calculations:
            params_desc = ', '.join(f"{k}={v}" for k, v in failure['params'].items())
            print(f"{failure['method']}: {params_desc}")
            error_lines = failure['error'].splitlines()
            print(f"  Error: {error_lines[0] if error_lines else '(no error message)'}")
            if 'traceback' in failure:
                print(f"  Full traceback:\n{failure['traceback']}")

    if show_plots:
        for energy_fig in figures:
            energy_fig.show()
        for fig_fill in filling_figures:
            fig_fill.show()

    return fig, results_payload


def compare_compressibility_fixed_supercluster(
    v_sep_ratio: Tuple[int, int],
    U_values: Sequence[float],
    V_values: Sequence[float],
    *,
    supercluster_int_seps: Optional[Dict[int, List[Tuple[int, Tuple[int, int]]]]] = None,
    supercluster_sizes: Optional[Sequence[int]] = None,
    max_seps: int = 3,
    n_mu_points: int = 30,
    mu_range_factor: float = 2.0,
    mu_min_range: float = 2.0,
    t: float = 1.0,
    L: int = 20,
    chi: int = 32,
    solver_method: str = 'dense_ED',
    states_retained: int = 4,
    output_dir: str = os.path.join(os.path.dirname(__file__), 'large_files', 'plots'),
    show_plots: bool = True,
    save_html: bool = True,
    save_data: bool = True,
    save_pdf: bool = True,
    filename_prefix: str = 'compressibility_fixed_supercluster',
    include_idmrg: bool = False,
    include_finite_dmrg: bool = True,
    results: Optional[Union[Dict, str, os.PathLike]] = None,
    axes: Tuple[str, str] = ('U', 'V'),
    rows_per_page: int = 4,
    cols_per_page: int = 3,
) -> Tuple[List[go.Figure], Dict]:
    """
    Compressibility plot for fixed supercluster sizes.

    Sweeps mu_0 on the x-axis and plots filling per site (n) on the y-axis.
    Each subplot shows a particular (row_param, SC_size) pair. Different
    (Nc, int_sep) combinations that produce the same supercluster size are
    shown as separate lines (maximal case in black).

    Grid layout:
    - Columns: supercluster sizes (always)
    - Rows: U or V values (controlled by axes[0])
    - Separate figure per value of the other param (axes[1])

    Args:
        v_sep_ratio: Ratio controlling the AA modulation for V.
        U_values: U values (Hubbard interaction strengths).
        V_values: V values (AA modulation strengths).
        supercluster_int_seps: Dict mapping sc_size -> list of (Nc, int_sep_ratio) pairs.
            If None, auto-generate using supercluster_sizes and max_seps.
        supercluster_sizes: List of target supercluster sizes (must divide L).
        n_mu_points: Number of mu_0 sweep points per (U, V) pair.
        mu_range_factor: mu_0 range = [-factor*(|U|+|V|), factor*(|U|+|V|)].
        mu_min_range: Minimum half-range for mu_0 when |U|+|V| is small.
        axes: ('U', 'V') -> rows=U, pages=V. ('V', 'U') -> rows=V, pages=U.
        Other args: Same as compare_fixed_supercluster.

    Returns:
        (list_of_figures, results_dict)
    """
    if t is None:
        raise ValueError("Parameter t must be specified.")

    def _coerce_ratio(value, label: str) -> Tuple[int, int]:
        if value is None:
            raise ValueError(f"{label} ratio must be provided.")
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{label} ratio must be a length-2 iterable, got {value!r}.")
        p = int(round(value[0]))
        q = int(round(value[1]))
        if q == 0:
            raise ValueError(f"Denominator for {label} ratio cannot be zero.")
        return (p, q)

    # Validate axes parameter
    if not isinstance(axes, (tuple, list)) or len(axes) != 2 or set(axes) != {'U', 'V'}:
        raise ValueError(f"axes must be ('U', 'V') or ('V', 'U'), got {axes!r}")
    row_param = axes[0]  # rows = this param

    v_sep_ratio = _coerce_ratio(v_sep_ratio, "V separation")

    U_arr = np.asarray(U_values, dtype=float)
    V_arr = np.asarray(V_values, dtype=float)
    if U_arr.ndim != 1 or U_arr.size == 0:
        raise ValueError("U_values must be a 1-D array with at least one entry.")
    if V_arr.ndim != 1 or V_arr.size == 0:
        raise ValueError("V_values must be a 1-D array with at least one entry.")
    u_list = [float(u) for u in U_arr]
    v_list = [float(v) for v in V_arr]

    from math import gcd

    # Auto-generate supercluster_int_seps if not provided
    if supercluster_int_seps is None:
        if supercluster_sizes is None:
            raise ValueError(
                "Either supercluster_int_seps or supercluster_sizes must be provided."
            )
        supercluster_int_seps = {}
        for sc_size in supercluster_sizes:
            if L % sc_size != 0:
                raise ValueError(f"supercluster_size={sc_size} must divide L={L}")
            valid_int_seps = compatible_int_seps(L, v_sep_ratio, sc_size)
            if not valid_int_seps:
                warnings.warn(f"No valid int_seps found for sc_size={sc_size}, skipping.")
                continue

            pairs: List[Tuple[int, Tuple[int, int]]] = []

            def is_trivial_int_sep(int_sep: Tuple[int, int]) -> bool:
                p, q = int_sep
                return p % q == 0

            valid_int_seps = [s for s in valid_int_seps if not is_trivial_int_sep(s)]

            maximal_int_sep = (1, sc_size)
            if maximal_int_sep in valid_int_seps:
                m = L // sc_size
                g1 = gcd(L, m)
                qm = L // g1
                if qm % sc_size == 0:
                    pairs.append((sc_size, maximal_int_sep))

            for int_sep in valid_int_seps:
                if len(pairs) >= max_seps:
                    break
                m = int(L * int_sep[0] / int_sep[1])
                g1 = gcd(L, m)
                qm = L // g1
                compatible_Nc_values = [nc for nc in range(2, qm + 1) if qm % nc == 0]
                for Nc in sorted(compatible_Nc_values, reverse=True):
                    if len(pairs) >= max_seps:
                        break
                    pair = (Nc, int_sep)
                    if pair not in pairs:
                        pairs.append(pair)

            if pairs:
                supercluster_int_seps[sc_size] = pairs
                print(f"Auto-generated for SC={sc_size}: {pairs}")

        if not supercluster_int_seps:
            raise ValueError("No valid supercluster configurations could be generated.")

    # Validate supercluster_int_seps
    sc_sizes = sorted(supercluster_int_seps.keys())
    if not sc_sizes:
        raise ValueError("supercluster_int_seps must have at least one entry.")

    validated_map: Dict[int, List[Tuple[int, Tuple[int, int]]]] = {}
    for sc_size, nc_int_sep_list in supercluster_int_seps.items():
        validated_map[sc_size] = []
        for entry in nc_int_sep_list:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(
                    f"Each entry in supercluster_int_seps[{sc_size}] must be (Nc, int_sep_ratio), got {entry!r}"
                )
            Nc = int(entry[0])
            int_sep = _coerce_ratio(entry[1], f"int_sep for sc_size={sc_size}, Nc={Nc}")
            computed_sc_size = compute_supercluster_size(L, int_sep, v_sep_ratio)
            if computed_sc_size != sc_size:
                raise ValueError(
                    f"(Nc={Nc}, int_sep={int_sep}) with v_sep={v_sep_ratio} gives supercluster_size={computed_sc_size}, "
                    f"not the claimed {sc_size}"
                )
            m = int(L * int_sep[0] / int_sep[1])
            g1 = gcd(L, m)
            qm = L // g1
            if qm % Nc != 0:
                raise ValueError(
                    f"Nc={Nc} is not compatible with int_sep={int_sep}: qm={qm} is not divisible by Nc={Nc}"
                )
            validated_map[sc_size].append((Nc, int_sep))

    # Row/page mapping
    if row_param == 'U':
        row_list = u_list
        page_list = v_list
        row_label_prefix = 'U'
        page_label_prefix = 'V'
    else:
        row_list = v_list
        page_list = u_list
        row_label_prefix = 'V'
        page_label_prefix = 'U'

    def _get_U_V(row_val: float, page_val: float) -> Tuple[float, float]:
        if row_param == 'U':
            return row_val, page_val
        return page_val, row_val

    # Generate per-(U, V) mu_0 arrays
    mu_arrays: Dict[Tuple[float, float], np.ndarray] = {}
    for U in u_list:
        for V in v_list:
            half_range = max(mu_range_factor * (abs(U) + abs(V)), mu_min_range)
            mu_arrays[(U, V)] = np.linspace(-half_range, half_range, n_mu_points)

    # Storage: cluster_fillings[(sc_size, Nc, int_sep, U, V)][mu_idx]
    cluster_fillings: Dict[Tuple[int, int, Tuple[int, int], float, float], np.ndarray]
    cluster_energies: Dict[Tuple[int, int, Tuple[int, int], float, float], np.ndarray]
    idmrg_fill_cache: Dict[Tuple[float, float], np.ndarray]
    finite_dmrg_fill_cache: Dict[Tuple[float, float], np.ndarray]
    failed_calculations: List[Dict] = []

    if results is not None:
        if isinstance(results, (str, os.PathLike)):
            results_path = Path(results)
            if not results_path.exists():
                raise ValueError(f"Results file not found: {results_path}")
            with open(results_path, 'rb') as fh:
                results = pickle.load(fh)
        elif not isinstance(results, dict):
            raise ValueError("results must be a dict or path-like object.")

        save_data = False
        print("Using precomputed results; skipping new simulations.")
        params = results.get('parameters', {})

        # Reconstruct mu_arrays
        stored_mu = results.get('mu_arrays', {})
        mu_arrays = {}
        for k, v in stored_mu.items():
            u_val, v_val = map(float, k.split('_'))
            mu_arrays[(u_val, v_val)] = np.asarray(v)

        # Load cluster data
        serialized_fills = results.get('cluster_fillings', {})
        serialized_energies = results.get('cluster_energies', {})
        cluster_fillings = {}
        cluster_energies = {}
        for sc_key, pairs_dict in serialized_fills.items():
            sc_size = int(sc_key)
            for pair_key, uv_dict in pairs_dict.items():
                parts = pair_key.split('_')
                Nc = int(parts[0])
                int_sep = (int(parts[1]), int(parts[2]))
                for uv_key, values in uv_dict.items():
                    u_val, v_val = map(float, uv_key.split('_'))
                    key = (sc_size, Nc, int_sep, u_val, v_val)
                    cluster_fillings[key] = np.asarray(values, dtype=float)

        for sc_key, pairs_dict in serialized_energies.items():
            sc_size = int(sc_key)
            for pair_key, uv_dict in pairs_dict.items():
                parts = pair_key.split('_')
                Nc = int(parts[0])
                int_sep = (int(parts[1]), int(parts[2]))
                for uv_key, values in uv_dict.items():
                    u_val, v_val = map(float, uv_key.split('_'))
                    key = (sc_size, Nc, int_sep, u_val, v_val)
                    cluster_energies[key] = np.asarray(values, dtype=float)

        # Load DMRG caches
        idmrg_fill_cache = {}
        finite_dmrg_fill_cache = {}
        stored_idmrg = results.get('idmrg_fillings', {})
        stored_finite = results.get('finite_dmrg_fillings', {})
        for U in u_list:
            for V in v_list:
                uv_key = f"{U}_{V}"
                n_mu = len(mu_arrays[(U, V)])
                idmrg_fill_cache[(U, V)] = np.asarray(stored_idmrg.get(uv_key, []), dtype=float) if stored_idmrg.get(uv_key) is not None else np.full(n_mu, np.nan)
                finite_dmrg_fill_cache[(U, V)] = np.asarray(stored_finite.get(uv_key, []), dtype=float) if stored_finite.get(uv_key) is not None else np.full(n_mu, np.nan)

        # Override metadata
        v_sep_ratio = _coerce_ratio(params.get('v_sep_ratio', v_sep_ratio), "V separation")
        t = params.get('t', t)
        L = params.get('L', L)
        chi = params.get('chi', chi)
        states_retained = params.get('states_retained', states_retained)
        include_idmrg = params.get('include_idmrg', include_idmrg)
        include_finite_dmrg = params.get('include_finite_dmrg', include_finite_dmrg)
    else:
        # Initialize storage
        cluster_fillings = {}
        cluster_energies = {}
        for sc_size in sc_sizes:
            for Nc, int_sep in validated_map[sc_size]:
                for U in u_list:
                    for V in v_list:
                        key = (sc_size, Nc, int_sep, U, V)
                        cluster_fillings[key] = np.full(n_mu_points, np.nan, dtype=float)
                        cluster_energies[key] = np.full(n_mu_points, np.nan, dtype=float)

        idmrg_fill_cache = {
            (U, V): np.full(n_mu_points, np.nan, dtype=float)
            for U in u_list for V in v_list
        }
        finite_dmrg_fill_cache = {
            (U, V): np.full(n_mu_points, np.nan, dtype=float)
            for U in u_list for V in v_list
        }

        print("=" * 60)
        print("Running compressibility (fixed supercluster) calculations")
        print(f"Supercluster sizes: {sc_sizes}")
        for sc_size in sc_sizes:
            for Nc, int_sep in validated_map[sc_size]:
                print(f"  SC={sc_size}: Nc={Nc}, int_sep={format_sep_as_pi(int_sep)}")
        print(f"v_sep = {format_sep_as_pi(v_sep_ratio)}")
        print(f"U values: {u_list}")
        print(f"V values: {v_list}")
        print(f"mu_0 points per (U, V): {n_mu_points}")
        print("=" * 60)

        # Compute cluster results
        total_calcs = sum(len(validated_map[sc]) for sc in sc_sizes) * len(u_list) * len(v_list) * n_mu_points
        with tqdm(total=total_calcs, desc="Cluster calcs", ncols=80) as pbar:
            for sc_size in sc_sizes:
                for Nc, int_sep in validated_map[sc_size]:
                    for V in v_list:
                        for U in u_list:
                            mu_arr = mu_arrays[(U, V)]
                            for mu_idx, mu_0 in enumerate(mu_arr):
                                physical_params = PhysicalParams(U=U, mu_0=mu_0, V=V, t=t)
                                run_config = ClusterModelConfig(
                                    L=L,
                                    int_cluster_size=Nc,
                                    cluster_separation_ratio=int_sep,
                                    V_separation_ratio=v_sep_ratio,
                                    ham_lib='quspin',
                                    physical_params=physical_params,
                                    model_bc='periodic',
                                    int_cluster_bc='periodic',
                                    super_cluster_bc='periodic',
                                    solver_method=solver_method,
                                    states_retained=states_retained,
                                )
                                try:
                                    system_expectations, _, _ = get_general_expectations(
                                        run_config, mu_eff=mu_0, return_mu=True,
                                    )
                                    energy, filling, _ = system_expectations
                                    key = (sc_size, Nc, int_sep, U, V)
                                    cluster_fillings[key][mu_idx] = filling / L
                                    cluster_energies[key][mu_idx] = (energy + mu_0 * filling) / L
                                except Exception as exc:
                                    import traceback
                                    failed_calculations.append({
                                        'method': f'cluster SC={sc_size} Nc={Nc}',
                                        'params': {'U': U, 'V': V, 'mu_0': mu_0},
                                        'error': str(exc) or repr(exc),
                                        'traceback': traceback.format_exc(),
                                    })
                                pbar.update(1)

        # DMRG reference sweeps
        print("\n" + "=" * 60)
        print("Computing DMRG references")
        print("=" * 60)

        for U in u_list:
            for V in v_list:
                mu_arr = mu_arrays[(U, V)]
                for mu_idx, mu_0 in enumerate(tqdm(mu_arr, desc=f"DMRG U={U} V={V}", ncols=80)):
                    if include_idmrg:
                        try:
                            _, filling_dmrg, _ = run_dmrg_method(
                                U, mu_0, V, v_sep_ratio, t, L, chi,
                            )
                            idmrg_fill_cache[(U, V)][mu_idx] = filling_dmrg
                        except Exception as exc:
                            import traceback
                            failed_calculations.append({
                                'method': 'iDMRG',
                                'params': {'U': U, 'V': V, 'mu_0': mu_0},
                                'error': str(exc),
                                'traceback': traceback.format_exc(),
                            })

                    if include_finite_dmrg:
                        try:
                            _, _, filling_finite = get_gnd(
                                L, chi, U, t, mu_0, V, v_sep_ratio,
                            )
                            finite_dmrg_fill_cache[(U, V)][mu_idx] = filling_finite
                        except Exception as exc:
                            import traceback
                            failed_calculations.append({
                                'method': 'Finite DMRG',
                                'params': {'U': U, 'V': V, 'mu_0': mu_0},
                                'error': str(exc),
                                'traceback': traceback.format_exc(),
                            })

    # --- COLOR MAPPING ---
    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf',
    ]

    def _get_pair_colors(sc_size: int, pairs: List[Tuple[int, Tuple[int, int]]]) -> Dict[Tuple[int, Tuple[int, int]], str]:
        """Black for maximal (Nc=SC, int_sep=(1,SC)), palette for others."""
        colors = {}
        non_maximal_idx = 0
        for Nc, int_sep in pairs:
            if Nc == sc_size and int_sep == (1, sc_size):
                colors[(Nc, int_sep)] = '#000000'
            else:
                colors[(Nc, int_sep)] = color_palette[non_maximal_idx % len(color_palette)]
                non_maximal_idx += 1
        return colors

    pair_color_maps = {sc: _get_pair_colors(sc, validated_map[sc]) for sc in sc_sizes}

    # --- PLOTTING ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)
    v_sep_label = format_sep_as_pi(v_sep_ratio)

    all_figures: List[go.Figure] = []
    html_paths: List[str] = []
    mpl_paths: List[str] = []

    # One figure per page_param value
    for page_val in page_list:
        n_rows_total = len(row_list)
        n_cols_total = len(sc_sizes)
        n_rows_pp = min(rows_per_page, n_rows_total)
        n_cols_pp = min(cols_per_page, n_cols_total)
        n_row_pages = int(np.ceil(n_rows_total / n_rows_pp))
        n_col_pages = int(np.ceil(n_cols_total / n_cols_pp))

        for rp in range(n_row_pages):
            for cp in range(n_col_pages):
                row_start = rp * n_rows_pp
                row_end = min(row_start + n_rows_pp, n_rows_total)
                col_start = cp * n_cols_pp
                col_end = min(col_start + n_cols_pp, n_cols_total)
                current_row_vals = row_list[row_start:row_end]
                current_sc_sizes = sc_sizes[col_start:col_end]
                n_rows_this = len(current_row_vals)
                n_cols_this = len(current_sc_sizes)

                subplot_titles = []
                for rv in current_row_vals:
                    for sc in current_sc_sizes:
                        subplot_titles.append(f"{row_label_prefix}={rv}, SC={sc}")

                fig = make_subplots(
                    rows=n_rows_this, cols=n_cols_this,
                    subplot_titles=subplot_titles,
                    horizontal_spacing=0.08,
                    vertical_spacing=0.12 / max(n_rows_this - 1, 1) if n_rows_this > 1 else 0.12,
                )

                legend_name_map = {}
                legend_idx = 1
                for r in range(1, n_rows_this + 1):
                    for c in range(1, n_cols_this + 1):
                        legend_name_map[(r, c)] = 'legend' if legend_idx == 1 else f'legend{legend_idx}'
                        legend_idx += 1

                any_trace = False
                for ri, row_val in enumerate(current_row_vals):
                    row = ri + 1
                    U, V = _get_U_V(row_val, page_val)
                    mu_arr = mu_arrays[(U, V)]

                    for ci, sc_size in enumerate(current_sc_sizes):
                        col = ci + 1

                        # DMRG reference
                        dmrg_color = '#DAA520'
                        if include_idmrg and np.any(np.isfinite(idmrg_fill_cache[(U, V)])):
                            fig.add_trace(
                                go.Scatter(
                                    x=mu_arr.tolist(), y=idmrg_fill_cache[(U, V)].tolist(),
                                    mode='lines+markers', name="iDMRG",
                                    legendgroup=f"iDMRG_{row}_{col}",
                                    marker=dict(color=dmrg_color, size=10, symbol='diamond'),
                                    line=dict(color=dmrg_color, width=2, dash='dot'),
                                    showlegend=True,
                                    hovertemplate="μ=%{x:.3g}<br>n_iDMRG=%{y:.4f}<extra></extra>",
                                    legend=legend_name_map[(row, col)],
                                ), row=row, col=col,
                            )
                            any_trace = True

                        if include_finite_dmrg and np.any(np.isfinite(finite_dmrg_fill_cache[(U, V)])):
                            fig.add_trace(
                                go.Scatter(
                                    x=mu_arr.tolist(), y=finite_dmrg_fill_cache[(U, V)].tolist(),
                                    mode='lines+markers', name="DMRG",
                                    legendgroup=f"finite_dmrg_{row}_{col}",
                                    marker=dict(color=dmrg_color, size=10, symbol='diamond-open'),
                                    line=dict(color=dmrg_color, width=2, dash='dash'),
                                    showlegend=True,
                                    hovertemplate="μ=%{x:.3g}<br>n_DMRG=%{y:.4f}<extra></extra>",
                                    legend=legend_name_map[(row, col)],
                                ), row=row, col=col,
                            )
                            any_trace = True

                        # Cluster lines for this SC size
                        pair_colors = pair_color_maps[sc_size]
                        for Nc, int_sep in validated_map[sc_size]:
                            key = (sc_size, Nc, int_sep, U, V)
                            fills = cluster_fillings.get(key)
                            if fills is None:
                                continue
                            trace_color = pair_colors[(Nc, int_sep)]
                            sep_label = format_sep_as_pi(int_sep)
                            is_maximal = (Nc == sc_size and int_sep == (1, sc_size))

                            xs = [mu_arr[i] for i in range(len(mu_arr)) if np.isfinite(fills[i])]
                            ys = [fills[i] for i in range(len(mu_arr)) if np.isfinite(fills[i])]
                            if not xs:
                                continue

                            fig.add_trace(
                                go.Scatter(
                                    x=xs, y=ys,
                                    mode='lines+markers',
                                    name=f"Nc={Nc}, m={sep_label}" + (" (max)" if is_maximal else ""),
                                    legendgroup=f"pair_{Nc}_{int_sep}_{row}_{col}",
                                    marker=dict(color=trace_color, size=6),
                                    line=dict(color=trace_color, width=2.5 if is_maximal else 1.5),
                                    showlegend=True,
                                    hovertemplate=f"Nc={Nc}<br>μ=%{{x:.3g}}<br>n=%{{y:.4f}}<extra></extra>",
                                    legend=legend_name_map[(row, col)],
                                ), row=row, col=col,
                            )
                            any_trace = True

                if not any_trace:
                    continue

                annotation_text = f"{page_label_prefix}={page_val}, v_sep={v_sep_label}, t={t}, L={L}, χ={chi}"
                legend_configs = {}
                for ri in range(n_rows_this):
                    for ci in range(n_cols_this):
                        lk = legend_name_map[(ri + 1, ci + 1)]
                        x_start = ci / n_cols_this
                        y_start = 1.0 - ri / n_rows_this
                        legend_configs[lk] = dict(
                            x=x_start + 0.01, y=y_start - 0.01,
                            xanchor='left', yanchor='top', font=dict(size=9),
                            bgcolor='rgba(255,255,255,0.85)', bordercolor='rgba(200,200,200,0.5)',
                        )

                fig.update_layout(
                    title_text=f"Compressibility: Fixed Supercluster<br><sub>{annotation_text}</sub>",
                    height=350 * n_rows_this + 100,
                    width=400 * n_cols_this,
                    margin=dict(b=80),
                    **legend_configs,
                )
                fig.update_xaxes(title_text="μ₀")
                fig.update_yaxes(title_text="n (filling/site)")

                if save_html:
                    page_tag = f"_{page_label_prefix}{page_val}"
                    html_name = f"{filename_prefix}_L{L}_chi{chi}_{timestamp}{page_tag}.html"
                    html_path = os.path.join(output_dir, html_name)
                    fig.write_html(html_path)
                    html_paths.append(html_path)
                    print(f"Saved figure to {html_path}")

                all_figures.append(fig)

    # --- MATPLOTLIB FIGURES ---
    if save_pdf:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        plt.rcParams.update({
            'font.family': 'serif', 'font.size': 10,
            'axes.labelsize': 12, 'axes.titlesize': 11,
            'legend.fontsize': 7,
            'xtick.labelsize': 9, 'ytick.labelsize': 9,
            'lines.linewidth': 1.5, 'lines.markersize': 4,
            'axes.linewidth': 0.8, 'grid.linewidth': 0.4, 'grid.alpha': 0.3,
            'figure.dpi': 150, 'savefig.dpi': 300,
            'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
        })
        try:
            plt.rcParams.update({
                'text.usetex': True,
                'text.latex.preamble': r'\usepackage{amsmath}',
            })
            _test_fig, _test_ax = plt.subplots(1, 1, figsize=(1, 1))
            _test_ax.set_xlabel(r"$\mu$")
            _test_fig.savefig(os.path.join(output_dir, "_latex_test.pdf"))
            plt.close(_test_fig)
            os.remove(os.path.join(output_dir, "_latex_test.pdf"))
            use_tex = True
        except Exception:
            plt.rcParams['text.usetex'] = False
            use_tex = False

        def _pi_label(sep_str: str) -> str:
            if use_tex:
                return '$' + sep_str.replace('π', r'\pi') + '$'
            return sep_str

        v_sep_latex = v_sep_label.replace('π', r'\pi') if use_tex else v_sep_label

        for page_val in page_list:
            mpl_n_rows = len(row_list)
            mpl_n_cols = len(sc_sizes)

            # Generate filling and relative error figures
            plot_modes = [False]
            if include_finite_dmrg:
                plot_modes.append(True)

            for is_rel_error in plot_modes:
                mode_tag = 'rel_error' if is_rel_error else 'filling'

                fig_width = 3.2 * mpl_n_cols + 0.6
                fig_height = 2.8 * mpl_n_rows + 0.8
                mpl_fig, axs = plt.subplots(
                    mpl_n_rows, mpl_n_cols,
                    figsize=(fig_width, fig_height),
                    squeeze=False, sharex=False,
                )

                REL_ERR_REF_THRESHOLD = 0.02

                for ri, row_val in enumerate(row_list):
                    U, V = _get_U_V(row_val, page_val)
                    mu_arr = mu_arrays[(U, V)]

                    for ci, sc_size in enumerate(sc_sizes):
                        ax = axs[ri, ci]
                        pair_colors = pair_color_maps[sc_size]

                        # DMRG reference (filling plot only)
                        dmrg_color = '#B8860B'
                        if not is_rel_error:
                            if include_idmrg and np.any(np.isfinite(idmrg_fill_cache[(U, V)])):
                                ax.plot(
                                    mu_arr, idmrg_fill_cache[(U, V)],
                                    color=dmrg_color, ls=':', lw=2.0,
                                    marker='D', ms=5, markerfacecolor='none', markeredgewidth=0.8,
                                    label='iDMRG', zorder=10,
                                )
                            if include_finite_dmrg and np.any(np.isfinite(finite_dmrg_fill_cache[(U, V)])):
                                ax.plot(
                                    mu_arr, finite_dmrg_fill_cache[(U, V)],
                                    color=dmrg_color, ls='--', lw=2.0,
                                    marker='D', ms=5, markerfacecolor='none', markeredgewidth=0.8,
                                    label='DMRG', zorder=10,
                                )

                        # Cluster lines
                        reference_series = finite_dmrg_fill_cache[(U, V)] if is_rel_error else None
                        for Nc, int_sep in validated_map[sc_size]:
                            key = (sc_size, Nc, int_sep, U, V)
                            fills = cluster_fillings.get(key)
                            if fills is None:
                                continue
                            trace_color = pair_colors[(Nc, int_sep)]
                            sep_label = format_sep_as_pi(int_sep)
                            is_maximal = (Nc == sc_size and int_sep == (1, sc_size))

                            xs, ys = [], []
                            for mu_idx, mu_0 in enumerate(mu_arr):
                                f_val = fills[mu_idx]
                                if not np.isfinite(f_val):
                                    continue
                                if is_rel_error and reference_series is not None:
                                    ref = reference_series[mu_idx]
                                    if not (np.isfinite(ref) and abs(ref) > REL_ERR_REF_THRESHOLD):
                                        continue
                                    xs.append(mu_0)
                                    ys.append(np.abs(f_val - ref) / np.abs(ref))
                                else:
                                    xs.append(mu_0)
                                    ys.append(f_val)

                            if not xs:
                                continue

                            nc_label = f'$N_c={Nc}$' if use_tex else f'Nc={Nc}'
                            max_tag = ' (max)' if is_maximal else ''
                            ax.plot(
                                xs, ys, color=trace_color, ls='-',
                                lw=2.0 if is_maximal else 1.3,
                                marker='o' if is_maximal else 's',
                                ms=4 if is_maximal else 3,
                                label=f'{nc_label}, m={_pi_label(sep_label)}{max_tag}',
                                zorder=5 if is_maximal else 3,
                            )

                        # Formatting
                        ax.set_xlabel(r'$\mu_0$' if use_tex else 'mu_0')
                        if ci == 0:
                            rlabel = f'${row_label_prefix}={row_val:g}$' if use_tex else f'{row_label_prefix}={row_val:g}'
                            if is_rel_error:
                                ylabel = r'Rel.\ error in $n$' if use_tex else 'Rel. error in n'
                            else:
                                ylabel = r'$n$ (filling/site)' if use_tex else 'n (filling/site)'
                            ax.set_ylabel(f'{rlabel}\n{ylabel}')
                        if ri == 0:
                            ax.set_title(f'SC = {sc_size}')

                        ax.grid(True, ls='--', alpha=0.3)
                        if not is_rel_error:
                            ax.set_ylim(-0.05, 2.05)
                        margin = 0.05 * (mu_arr[-1] - mu_arr[0]) if len(mu_arr) > 1 else 0.5
                        ax.set_xlim(mu_arr[0] - margin, mu_arr[-1] + margin)
                        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

                        ax.legend(
                            loc='upper right', framealpha=0.85,
                            edgecolor='0.7', handlelength=1.5,
                            borderpad=0.3, labelspacing=0.25, fontsize=6,
                        )

                # Suptitle
                page_info = f'{page_label_prefix}={page_val}'
                if is_rel_error:
                    suptitle = (
                        f'Rel.~Error: {page_info}, $v_{{\\mathrm{{sep}}}}={v_sep_latex}$, $t={t}$, $L={L}$, $\\chi={chi}$'
                        if use_tex else
                        f'Rel. Error: {page_info}, v_sep={v_sep_label}, t={t}, L={L}, chi={chi}'
                    )
                else:
                    suptitle = (
                        f'Compressibility (Fixed SC): {page_info}, $v_{{\\mathrm{{sep}}}}={v_sep_latex}$, $t={t}$, $L={L}$, $\\chi={chi}$'
                        if use_tex else
                        f'Compressibility (Fixed SC): {page_info}, v_sep={v_sep_label}, t={t}, L={L}, chi={chi}'
                    )
                mpl_fig.suptitle(suptitle, fontsize=13, y=1.01)
                mpl_fig.tight_layout(rect=[0, 0, 0.96, 1.0])

                for ext in ('pdf', 'svg'):
                    page_tag = f"_{page_label_prefix}{page_val}"
                    mpl_name = f"{filename_prefix}_{mode_tag}_L{L}_chi{chi}_{timestamp}{page_tag}.{ext}"
                    mpl_path = os.path.join(output_dir, mpl_name)
                    mpl_fig.savefig(mpl_path, format=ext)
                    mpl_paths.append(mpl_path)
                    print(f"Saved matplotlib figure to {mpl_path}")

                if show_plots:
                    plt.show()
                else:
                    plt.close(mpl_fig)

    # --- SERIALIZATION ---
    saved_paths: Dict[str, Any] = {}
    if save_html:
        saved_paths['html_pages'] = html_paths
        if html_paths:
            saved_paths['html'] = html_paths[0]
    if mpl_paths:
        saved_paths['mpl_figures'] = mpl_paths

    # Serialize cluster data: {sc_key: {pair_key: {uv_key: [values]}}}
    cluster_fillings_serialized: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    cluster_energies_serialized: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for sc_size in sc_sizes:
        sc_key = str(sc_size)
        cluster_fillings_serialized[sc_key] = {}
        cluster_energies_serialized[sc_key] = {}
        for Nc, int_sep in validated_map[sc_size]:
            pair_key = f"{Nc}_{int_sep[0]}_{int_sep[1]}"
            cluster_fillings_serialized[sc_key][pair_key] = {}
            cluster_energies_serialized[sc_key][pair_key] = {}
            for U in u_list:
                for V in v_list:
                    uv_key = f"{U}_{V}"
                    key = (sc_size, Nc, int_sep, U, V)
                    cluster_fillings_serialized[sc_key][pair_key][uv_key] = cluster_fillings[key].tolist()
                    cluster_energies_serialized[sc_key][pair_key][uv_key] = cluster_energies[key].tolist()

    mu_arrays_serialized = {f"{U}_{V}": mu_arrays[(U, V)].tolist() for U in u_list for V in v_list}
    idmrg_serialized = {f"{U}_{V}": idmrg_fill_cache[(U, V)].tolist() for U in u_list for V in v_list}
    finite_dmrg_serialized = {f"{U}_{V}": finite_dmrg_fill_cache[(U, V)].tolist() for U in u_list for V in v_list}

    # Serialize validated_map
    sc_map_serialized = {}
    for sc_size, pairs in validated_map.items():
        sc_map_serialized[str(sc_size)] = [(Nc, list(int_sep)) for Nc, int_sep in pairs]

    results_payload = {
        'supercluster_sizes': sc_sizes,
        'supercluster_int_seps': sc_map_serialized,
        'U_values': u_list,
        'V_values': v_list,
        'mu_arrays': mu_arrays_serialized,
        'cluster_fillings': cluster_fillings_serialized,
        'cluster_energies': cluster_energies_serialized,
        'idmrg_fillings': idmrg_serialized,
        'finite_dmrg_fillings': finite_dmrg_serialized,
        'parameters': {
            'v_sep_ratio': v_sep_ratio,
            't': t, 'L': L, 'chi': chi,
            'solver_method': solver_method,
            'states_retained': states_retained,
            'include_idmrg': include_idmrg,
            'include_finite_dmrg': include_finite_dmrg,
            'n_mu_points': n_mu_points,
            'mu_range_factor': mu_range_factor,
            'mu_min_range': mu_min_range,
        },
        'artifacts': saved_paths,
        'failures': failed_calculations,
    }

    if save_data:
        pickle_name = f"{filename_prefix}_L{L}_chi{chi}_{timestamp}.pkl"
        pickle_path = os.path.join(output_dir, pickle_name)
        with open(pickle_path, 'wb') as fh:
            pickle.dump(results_payload, fh)
        saved_paths['pickle'] = pickle_path
        print(f"Saved data to {pickle_path}")

    if failed_calculations:
        print("\n" + "=" * 60)
        print("WARNING: Some calculations failed")
        print("=" * 60)
        for failure in failed_calculations:
            params_desc = ', '.join(f"{k}={v}" for k, v in failure['params'].items())
            print(f"{failure['method']}: {params_desc}")
            error_lines = failure['error'].splitlines()
            print(f"  Error: {error_lines[0] if error_lines else '(no error message)'}")

    if show_plots:
        for f in all_figures:
            f.show()

    return all_figures, results_payload


# =============================================================================
# Plotting-only: compare half/quarter filling for fixed supercluster
# =============================================================================


def plot_supercluster_filling_comparison(
    results_half: Union[str, os.PathLike],
    results_quarter: Union[str, os.PathLike],
    *,
    U_fixed: float,
    output_dir: str = 'large_files/plots',
    show_plots: bool = True,
    save_html: bool = True,
    filename_prefix: str = 'supercluster_filling_comparison',
    plot_relative_error: bool = False,
    log_yaxis: bool = True,
    shared_yaxis: bool = False,
) -> Tuple[go.Figure, Dict]:
    """
    Plot energy vs V for two fillings side-by-side (no computation).

    Loads two pre-computed result pickles (from ``compare_fixed_supercluster``)
    and creates a 2-row subplot grid:

    - Top row: half-filling results
    - Bottom row: quarter-filling results
    - Columns: supercluster sizes (intersection of both result sets)
    - X-axis: V values at a single fixed U
    - Traces: (Nc, int_sep) pairs + DMRG reference

    Args:
        results_half: Path to pickle from ``compare_fixed_supercluster`` at half filling.
        results_quarter: Path to pickle at quarter filling.
        U_fixed: The U value to slice out of each results grid.
        output_dir: Directory for saved HTML.
        show_plots: Whether to display the figure.
        save_html: Whether to save an interactive HTML.
        filename_prefix: Prefix for saved file names.
        plot_relative_error: If True, plot |E_cluster - E_dmrg| / |E_dmrg|.
        log_yaxis: Use log y-axis when plotting relative error.
        shared_yaxis: If True, all columns in the same row share the same y-axis range.

    Returns:
        (figure, info_dict)
    """

    # ------------------------------------------------------------------
    # Load pickles
    # ------------------------------------------------------------------
    def _load(path: Union[str, os.PathLike]) -> Dict:
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    data_half = _load(results_half)
    data_quarter = _load(results_quarter)

    filling_rows: List[Tuple[str, Dict]] = [
        (f"n={data_half.get('set_filling', 1.0)}", data_half),
        (f"n={data_quarter.get('set_filling', 0.5)}", data_quarter),
    ]

    # ------------------------------------------------------------------
    # Validate & resolve shared structure
    # ------------------------------------------------------------------
    sc_half = set(data_half['supercluster_sizes'])
    sc_quarter = set(data_quarter['supercluster_sizes'])
    sc_sizes = sorted(sc_half & sc_quarter)
    if not sc_sizes:
        raise ValueError(
            f"No shared supercluster sizes between the two results: "
            f"{sorted(sc_half)} vs {sorted(sc_quarter)}"
        )

    v_list_half = data_half['V_values']
    v_list_quarter = data_quarter['V_values']
    if v_list_half != v_list_quarter:
        raise ValueError(
            "V_values differ between the two results files; "
            "they must match for side-by-side comparison."
        )
    v_list = v_list_half

    # Find u_idx in each
    def _find_u_idx(data: Dict, U: float) -> int:
        u_vals = data['U_values']
        for i, u in enumerate(u_vals):
            if abs(u - U) < 1e-12:
                return i
        raise ValueError(
            f"U_fixed={U} not found in results U_values={u_vals}"
        )

    u_idx_half = _find_u_idx(data_half, U_fixed)
    u_idx_quarter = _find_u_idx(data_quarter, U_fixed)
    u_indices = [u_idx_half, u_idx_quarter]

    # ------------------------------------------------------------------
    # Subplot grid: 2 rows (fillings) x N_SC cols
    # ------------------------------------------------------------------
    n_rows = 2
    n_cols = len(sc_sizes)

    subplot_titles = []
    for fill_label, _ in filling_rows:
        for sc_size in sc_sizes:
            subplot_titles.append(f"{fill_label}, SC={sc_size}")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # Per-subplot legends
    legend_name_map: Dict[Tuple[int, int], str] = {}
    legend_idx = 1
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            if legend_idx == 1:
                legend_name_map[(r, c)] = 'legend'
            else:
                legend_name_map[(r, c)] = f'legend{legend_idx}'
            legend_idx += 1

    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf',
    ]
    dmrg_color = '#DAA520'

    # ------------------------------------------------------------------
    # Pre-compute shared y-axis ranges per row (if requested)
    # ------------------------------------------------------------------
    shared_y_ranges: Dict[int, Optional[List[float]]] = {0: None, 1: None}
    if shared_yaxis:
        for row_idx, (_, data) in enumerate(filling_rows):
            u_idx = u_indices[row_idx]
            int_seps_map = data['supercluster_int_seps']
            cluster_energies = data['cluster_energies']
            dmrg_energies = np.array(data.get('finite_dmrg_energies', []))
            include_dmrg = dmrg_energies.size > 0

            row_ys: List[float] = []
            for sc_size in sc_sizes:
                sc_key = str(sc_size)
                # DMRG values
                if include_dmrg and not plot_relative_error:
                    for v_idx in range(len(v_list)):
                        val = dmrg_energies[v_idx, u_idx]
                        if np.isfinite(val):
                            row_ys.append(float(val))
                # Cluster values
                raw_pairs = int_seps_map.get(sc_key, [])
                for entry in raw_pairs:
                    Nc = int(entry[0])
                    int_sep = (int(entry[1][0]), int(entry[1][1]))
                    pair_key = f"{Nc}_{int_sep[0]}_{int_sep[1]}"
                    sc_en = cluster_energies.get(sc_key, {})
                    if pair_key not in sc_en:
                        continue
                    energy_grid = sc_en[pair_key]
                    for v_idx in range(len(v_list)):
                        c_en = energy_grid[v_idx][u_idx]
                        if not np.isfinite(c_en):
                            continue
                        if plot_relative_error:
                            if include_dmrg:
                                ref = dmrg_energies[v_idx, u_idx]
                                if np.isfinite(ref) and ref != 0:
                                    row_ys.append(abs(c_en - ref) / abs(ref))
                        else:
                            row_ys.append(c_en)

            if row_ys:
                y_min, y_max = min(row_ys), max(row_ys)
                margin = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
                shared_y_ranges[row_idx] = [y_min - margin, y_max + margin]

    # ------------------------------------------------------------------
    # Plot traces
    # ------------------------------------------------------------------
    any_trace = False

    for row_idx, (fill_label, data) in enumerate(filling_rows):
        row = row_idx + 1
        u_idx = u_indices[row_idx]
        int_seps_map = data['supercluster_int_seps']  # {str(sc): [[Nc,[p,q]], ...]}
        cluster_energies = data['cluster_energies']
        dmrg_energies = np.array(data.get('finite_dmrg_energies', []))
        include_dmrg = dmrg_energies.size > 0

        for col_idx, sc_size in enumerate(sc_sizes):
            col = col_idx + 1
            sc_key = str(sc_size)

            # Parse (Nc, int_sep) list for this SC size
            raw_pairs = int_seps_map.get(sc_key, [])
            nc_int_sep_list: List[Tuple[int, Tuple[int, int]]] = [
                (int(entry[0]), (int(entry[1][0]), int(entry[1][1])))
                for entry in raw_pairs
            ]

            # Color map: true maximal in black
            pair_color_map: Dict[Tuple[int, Tuple[int, int]], str] = {}
            non_maximal: List[Tuple[int, Tuple[int, int]]] = []
            true_maximal = (sc_size, (1, sc_size))
            for Nc, int_sep in nc_int_sep_list:
                if (Nc, int_sep) == true_maximal:
                    pair_color_map[(Nc, int_sep)] = '#000000'
                else:
                    non_maximal.append((Nc, int_sep))
            for idx, pair in enumerate(non_maximal):
                pair_color_map[pair] = color_palette[idx % len(color_palette)]

            # --- DMRG reference ---
            if include_dmrg and not plot_relative_error:
                dmrg_xs: List[float] = []
                dmrg_ys: List[float] = []
                for v_idx, V in enumerate(v_list):
                    val = dmrg_energies[v_idx, u_idx]
                    if np.isfinite(val):
                        dmrg_xs.append(V)
                        dmrg_ys.append(float(val))
                if dmrg_xs:
                    fig.add_trace(
                        go.Scatter(
                            x=dmrg_xs,
                            y=dmrg_ys,
                            mode='lines+markers',
                            name="DMRG",
                            legendgroup=f"DMRG_{row}_{col}",
                            marker=dict(color=dmrg_color, size=10, symbol='diamond'),
                            line=dict(color=dmrg_color, width=2, dash='dot'),
                            showlegend=True,
                            hovertemplate="DMRG<br>V=%{x}<br>E=%{y:.6f}<extra></extra>",
                            legend=legend_name_map[(row, col)],
                        ),
                        row=row,
                        col=col,
                    )
                    any_trace = True

            # --- Cluster traces ---
            sc_energies = cluster_energies.get(sc_key, {})
            for Nc, int_sep in nc_int_sep_list:
                pair_key = f"{Nc}_{int_sep[0]}_{int_sep[1]}"
                if pair_key not in sc_energies:
                    continue

                energy_grid = sc_energies[pair_key]  # 2D list [v_idx][u_idx]
                xs: List[float] = []
                ys: List[float] = []
                hover_text: List[str] = []
                for v_idx, V in enumerate(v_list):
                    c_en = energy_grid[v_idx][u_idx]
                    if not np.isfinite(c_en):
                        continue

                    if plot_relative_error:
                        if not include_dmrg:
                            continue
                        ref = dmrg_energies[v_idx, u_idx]
                        if not (np.isfinite(ref) and ref != 0):
                            continue
                        c_err = abs(c_en - ref) / abs(ref)
                        xs.append(V)
                        ys.append(c_err)
                        hover_text.append(
                            f"V={V}<br>Nc={Nc}, int_sep={int_sep}<br>rel_err={c_err:.2%}"
                        )
                    else:
                        xs.append(V)
                        ys.append(c_en)
                        hover_text.append(
                            f"V={V}<br>Nc={Nc}, int_sep={int_sep}<br>E={c_en:.6f}"
                        )

                if not xs:
                    continue

                trace_color = pair_color_map.get((Nc, int_sep), color_palette[0])
                int_sep_label = format_sep_as_pi(int_sep)
                is_true_maximal = (Nc == sc_size and int_sep == (1, sc_size))
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode='lines+markers',
                        name=f"Nc={Nc}, m={int_sep_label}" + (" (max)" if is_true_maximal else ""),
                        legendgroup=f"pair_{Nc}_{int_sep}_{row}_{col}",
                        marker=dict(color=trace_color, size=8),
                        line=dict(color=trace_color, width=2),
                        showlegend=True,
                        hovertemplate="%{text}<extra></extra>",
                        text=hover_text,
                        legend=legend_name_map[(row, col)],
                    ),
                    row=row,
                    col=col,
                )
                any_trace = True

            # Axis labels
            fig.update_xaxes(title_text="V", row=row, col=col)
            y_title = "Relative error" if plot_relative_error else "Energy per site"
            y_kwargs: Dict[str, Any] = dict(
                title_text=y_title,
                type='log' if (plot_relative_error and log_yaxis) else 'linear',
            )
            if plot_relative_error:
                y_kwargs['tickformat'] = '.1%'
            if shared_yaxis and shared_y_ranges[row_idx] is not None:
                y_range = shared_y_ranges[row_idx]
                if plot_relative_error and log_yaxis:
                    lo = max(y_range[0], 1e-15)
                    hi = max(y_range[1], 1e-15)
                    y_kwargs['range'] = [np.log10(lo), np.log10(hi)]
                else:
                    y_kwargs['range'] = y_range
            fig.update_yaxes(row=row, col=col, **y_kwargs)

    if not any_trace:
        warnings.warn("No valid data found in either results file.")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    params_half = data_half.get('parameters', {})
    v_sep_ratio = params_half.get('v_sep_ratio', (0, 1))
    v_sep_label = format_sep_as_pi(tuple(v_sep_ratio))
    annotation_parts = [
        f"U={U_fixed}",
        f"v_sep={v_sep_label}",
        f"t={params_half.get('t', '?')}",
        f"L={params_half.get('L', '?')}",
        f"chi={params_half.get('chi', '?')}",
    ]
    annotation_text = ", ".join(annotation_parts)

    # Legend positioning
    h_spacing = 0.08
    v_spacing = 0.12
    subplot_width = (1.0 - h_spacing * (n_cols - 1)) / n_cols
    subplot_height = (1.0 - v_spacing * (n_rows - 1)) / n_rows

    legend_configs: Dict[str, Dict] = {}
    for r_idx in range(n_rows):
        for c_idx in range(n_cols):
            r = r_idx + 1
            c = c_idx + 1
            legend_key = legend_name_map[(r, c)]
            x_end = c_idx * (subplot_width + h_spacing) + subplot_width
            y_start = 1.0 - r_idx * (subplot_height + v_spacing) - subplot_height
            legend_configs[legend_key] = dict(
                x=x_end - 0.01,
                y=y_start + 0.02,
                xanchor='right',
                yanchor='bottom',
                bgcolor='rgba(255, 255, 255, 0.85)',
                bordercolor='rgba(0, 0, 0, 0.3)',
                borderwidth=1,
                font=dict(size=8),
                itemsizing='constant',
                tracegroupgap=0,
                itemwidth=30,
            )

    fig.update_layout(
        title=dict(
            text=(
                f"Relative Error vs V at U={U_fixed} (Fixed Supercluster Size)"
                if plot_relative_error
                else f"Ground State Energy vs V at U={U_fixed} (Fixed Supercluster Size)"
            ),
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top',
        ),
        hovermode='closest',
        height=350 * n_rows,
        width=400 * n_cols,
        margin=dict(b=80),
        **legend_configs,
    )
    fig.add_annotation(
        text=annotation_text,
        x=0.5,
        xref='paper',
        y=-0.08,
        yref='paper',
        showarrow=False,
        font=dict(size=11, color='gray'),
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_paths: Dict[str, str] = {}

    if save_html:
        L = params_half.get('L', '')
        chi = params_half.get('chi', '')
        html_name = f"{filename_prefix}_U{U_fixed}_L{L}_chi{chi}_{timestamp}.html"
        html_path = os.path.join(output_dir, html_name)
        fig.write_html(html_path)
        saved_paths['html'] = html_path
        print(f"Saved figure to {html_path}")

    if show_plots:
        fig.show()

    info = {
        'U_fixed': U_fixed,
        'supercluster_sizes': sc_sizes,
        'V_values': v_list,
        'filling_labels': [label for label, _ in filling_rows],
        'artifacts': saved_paths,
    }
    return fig, info
