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
from typing import Tuple, List, Dict, Optional, Sequence, Union
import os
import warnings
import pickle
from datetime import datetime
from pathlib import Path

from aah_code.cluster_model.model import ClusterModelConfig, PhysicalParams
from aah_code.cluster_model.clustering import generate_clusters
from aah_code.cluster_model.run_scripts_me import get_general_expectations
from aah_code.real_space_dmrg import run_dmrg_method, get_gnd, get_gnd_fixed_filling

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
        return f'{frac.numerator}π'
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
                    failed_calculations.append({
                        'method': ref_label,
                        'params': {'U': U, 'V': V},
                        'error': str(exc),
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
                        failed_calculations.append({
                            'method': f'cluster Nc={Nc}',
                            'params': {'U': U, 'V': V},
                            'error': str(exc),
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
            print(f"  Error: {failure['error'].splitlines()[0]}")

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
                        failed_calculations.append({
                            'method': f'cluster mu (Nc={mu_source_Nc})',
                            'params': {'U': U, 'V': V},
                            'error': str(exc),
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
                        failed_calculations.append({
                            'method': f'cluster Nc={Nc}',
                            'params': {'U': U, 'V': V},
                            'error': str(exc),
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
                        failed_calculations.append({
                            'method': 'iDMRG',
                            'params': {'U': U, 'V': V},
                            'error': str(exc),
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
                        failed_calculations.append({
                            'method': 'Finite DMRG',
                            'params': {'U': U, 'V': V},
                            'error': str(exc),
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
            print(f"  Error: {failure['error'].splitlines()[0]}")

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
    dmrg_fixed_filling: bool = False,
    results: Optional[Union[Dict, str, os.PathLike]] = None,
) -> Tuple[go.Figure, Dict]:
    """
    Compare half-filling and quarter-filling results across cluster sizes and interaction separations.

    Creates a 2x3 grid per page where:
    - Top row: Half-filling (mu_0 = U/2) for each cluster size
    - Bottom row: Quarter-filling (mu_0 = 0) for the same cluster sizes
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
            When None (default), top row uses mu_0=U/2 (half-filling) and bottom row uses mu_0=0.
        dmrg_fixed_filling: If True, finite DMRG uses canonical ensemble (fixed N).
        results: Pre-computed results dict or path to pickle file to load instead of computing.

    Returns:
        (figure, results_dict)
    """
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

    # Define the two filling modes: half-filling (mu_0=U/2) and quarter-filling (mu_0=0)
    # When set_filling is provided, it overrides both rows with that target filling
    filling_modes = ['half', 'quarter']  # top row, bottom row

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
                        # Determine mu_0 based on filling mode
                        if set_filling is not None:
                            # Use set_filling for both rows
                            mu0 = U / 2.0  # Initial guess, will be adjusted by set_filling
                            target_filling = set_filling
                        else:
                            if fill_mode == 'half':
                                mu0 = U / 2.0
                                target_filling = None
                            else:  # quarter
                                mu0 = 0.0
                                target_filling = None

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
                            }

                            if target_filling is not None:
                                system_expectations, _, mu_eff = time_call(
                                    timing_recorder,
                                    meta,
                                    get_general_expectations,
                                    run_config,
                                    timing_recorder=timing_recorder,
                                    set_filling=target_filling,
                                    return_mu=True,
                                )
                            else:
                                system_expectations, _ = time_call(
                                    timing_recorder,
                                    meta,
                                    get_general_expectations,
                                    run_config,
                                    timing_recorder=timing_recorder,
                                    mu_eff=mu0,
                                )

                            energy, filling, _ = system_expectations
                            energy_subtracted = (energy + mu0 * filling) / L
                            key = (Nc, int_sep, fill_mode)
                            cluster_results[key][u_idx] = energy_subtracted
                            cluster_fillings[key][u_idx] = filling / L
                        except Exception as exc:
                            failed_calculations.append({
                                'method': f'cluster Nc={Nc}, int_sep={format_sep_as_pi(int_sep)}, {fill_mode}',
                                'params': {'U': U, 'V': V},
                                'error': str(exc),
                            })

        # Compute DMRG references for each filling mode
        print("\n")
        print("=" * 60)
        print("Computing reference energies")
        print("=" * 60)

        for fill_idx, fill_mode in enumerate(filling_modes):
            for u_idx, U in enumerate(u_list):
                if set_filling is not None:
                    mu_eff_value = U / 2.0  # This would need proper mu solving for DMRG
                    target_fill = set_filling
                else:
                    if fill_mode == 'half':
                        mu_eff_value = U / 2.0
                        target_fill = 0.5
                    else:
                        mu_eff_value = 0.0
                        target_fill = None  # Grand canonical with mu=0

                # iDMRG
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
                        failed_calculations.append({
                            'method': f'iDMRG ({fill_mode})',
                            'params': {'U': U, 'V': V},
                            'error': str(exc),
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
                        if dmrg_fixed_filling and target_fill is not None:
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
                        failed_calculations.append({
                            'method': f'Finite DMRG ({fill_mode})',
                            'params': {'U': U, 'V': V},
                            'error': str(exc),
                        })

    # --- PLOTTING LOGIC ---
    color_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf',
    ]
    int_sep_color_map = {int_sep: color_palette[idx % len(color_palette)] for idx, int_sep in enumerate(all_int_sep_ratios)}

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with 2 rows (half/quarter filling) x N_c columns
    n_cols = len(cluster_sizes)
    n_cols_per_page = 3
    n_pages = int(np.ceil(n_cols / n_cols_per_page))

    figures: List[go.Figure] = []
    html_paths: List[str] = []

    for page_idx in range(n_pages):
        start_col = page_idx * n_cols_per_page
        end_col = min(start_col + n_cols_per_page, n_cols)
        current_cluster_sizes = cluster_sizes[start_col:end_col]
        n_cols_this_page = len(current_cluster_sizes)

        # Subplot titles
        top_titles = [f"Half-filling (μ₀=U/2): N_c={Nc}" for Nc in current_cluster_sizes]
        bottom_titles = [f"Quarter-filling (μ₀=0): N_c={Nc}" for Nc in current_cluster_sizes]

        fig = make_subplots(
            rows=2,
            cols=n_cols_this_page,
            subplot_titles=top_titles + bottom_titles,
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )

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

                # Add DMRG reference lines
                idmrg_energies = idmrg_cache[:, row_idx]
                if np.any(np.isfinite(idmrg_energies)) and not plot_relative_error:
                    fig.add_trace(
                        go.Scatter(
                            x=u_list,
                            y=idmrg_energies,
                            mode='lines+markers',
                            name="iDMRG",
                            legendgroup="iDMRG",
                            marker=dict(color='black', size=6, symbol='x'),
                            line=dict(color='black', width=2, dash='dash'),
                            showlegend=(col_idx == 0 and row_idx == 0),
                            hovertemplate="U=%{x:.3g}<br>E_iDMRG=%{y:.6f}<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )
                    any_trace = True

                finite_dmrg_energies = finite_dmrg_cache[:, row_idx]
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
                            showlegend=(col_idx == 0 and row_idx == 0),
                            hovertemplate="U=%{x:.3g}<br>E_Finite=%{y:.6f}<extra></extra>",
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
                    fig.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode='lines+markers',
                            name=f"m={int_sep_label}",
                            legendgroup=f"int_sep_{int_sep[0]}_{int_sep[1]}",
                            marker=dict(color=int_sep_color_map[int_sep], size=8),
                            line=dict(color=int_sep_color_map[int_sep], width=2),
                            showlegend=(col_idx == 0 and row_idx == 0),
                            hovertemplate="%{text}<extra></extra>",
                            text=hover_text,
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

        fig.update_layout(
            title=dict(
                text="Relative Error vs U (Half vs Quarter Filling)" if plot_relative_error
                     else "Ground State Energy vs U (Half vs Quarter Filling)",
                x=0.5,
                xanchor='center'
            ),
            hovermode='closest',
            legend_title="Int. Separation",
            height=700,
            width=400 * n_cols_this_page,
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
            page_suffix = f"_page_{page_idx + 1}" if n_pages > 1 else ""
            html_name = f"{filename_prefix}_L{L}_chi{chi}_{timestamp}{page_suffix}.html"
            html_path = os.path.join(output_dir, html_name)
            fig.write_html(html_path)
            html_paths.append(html_path)
            print(f"Saved figure to {html_path}")

        figures.append(fig)

    fig = figures[0] if figures else None
    saved_paths = {}
    if save_html:
        saved_paths['html_pages'] = html_paths
        if html_paths:
            saved_paths['html'] = html_paths[0]

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
            print(f"  Error: {failure['error'].splitlines()[0]}")

    if show_plots and fig is not None:
        fig.show()

    return fig, results_payload
