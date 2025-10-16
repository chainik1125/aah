"""
Plotting functions for comparing different cluster model setups with iDMRG.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Tuple, List
import os
import pickle
from datetime import datetime

from aah_code.cluster_model.model import ClusterModelConfig, PhysicalParams
from aah_code.cluster_model.run_scripts_me import get_general_expectations
from aah_code.real_space_dmrg import run_dmrg_method, get_gnd

# Configure plotly to work outside of notebooks
pio.renderers.default = "browser"

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable


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
    include_finite_dmrg: bool = False
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
                    energy_idmrg, filling_idmrg, _ = run_dmrg_method(U, mu_0, V, v_sep_ratio, t, L, chi)
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
                    energy_finite, _, filling_finite = get_gnd(L, chi, U, t, mu_0, V, v_sep_ratio)
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
                    
                    system_expectations, _ = get_general_expectations(run_config)
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
    
    # Helper function to format separation as fraction of π
    def format_sep_as_pi(sep_tuple):
        """Convert separation ratio to π fraction notation."""
        numerator = 2 * sep_tuple[0]
        denominator = sep_tuple[1]
        if numerator == denominator:
            return 'π'
        elif numerator == 1:
            return f'π/{denominator}'
        else:
            from fractions import Fraction
            frac = Fraction(numerator, denominator)
            if frac.denominator == 1:
                return f'{frac.numerator}π'
            return f'{frac.numerator}π/{frac.denominator}'
    
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