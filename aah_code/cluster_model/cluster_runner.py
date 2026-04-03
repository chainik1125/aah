"""
CLI runner for cluster comparison functions.

Usage:
    python -m aah_code.cluster_model.cluster_runner <command> [options]
    python -m aah_code.cluster_model.cluster_runner --config=config.yaml

Commands:
    filling_cluster_sizes   - Compare filling across cluster sizes
    fixed_supercluster      - Compare fixed supercluster configurations

Examples:
    # Run filling_cluster_sizes with default params
    python -m aah_code.cluster_model.cluster_runner filling_cluster_sizes

    # Run with custom parameters
    python -m aah_code.cluster_model.cluster_runner filling_cluster_sizes --L=48 --chi=64

    # Run fixed_supercluster at quarter filling
    python -m aah_code.cluster_model.cluster_runner fixed_supercluster --set_filling=0.5

    # Run from YAML config file
    python -m aah_code.cluster_model.cluster_runner --config=runs/my_run.yaml

    # Load from existing results file
    python -m aah_code.cluster_model.cluster_runner filling_cluster_sizes --results=path/to/file.pkl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any


def parse_int_list(s: str) -> List[int]:
    """Parse comma-separated integers: '2,4,6,8' -> [2, 4, 6, 8]"""
    return [int(x.strip()) for x in s.split(',')]


def parse_float_list(s: str) -> List[float]:
    """Parse comma-separated floats: '0,1,2,3' -> [0.0, 1.0, 2.0, 3.0]"""
    return [float(x.strip()) for x in s.split(',')]


def parse_ratio(s: str) -> tuple:
    """Parse ratio string: '1,2' -> (1, 2)"""
    parts = s.split(',')
    if len(parts) != 2:
        raise ValueError(f"Ratio must be 'p,q' format, got: {s}")
    return (int(parts[0].strip()), int(parts[1].strip()))


def parse_int_sep_ratios(s: str) -> dict:
    """
    Parse int_sep_ratios dict from string.
    Format: 'Nc1:p1,q1;Nc2:p2,q2' -> {Nc1: (p1, q1), Nc2: (p2, q2)}
    Example: '2:1,2;4:1,4;6:1,6' -> {2: (1,2), 4: (1,4), 6: (1,6)}
    """
    result = {}
    for pair in s.split(';'):
        if ':' not in pair:
            continue
        nc_str, ratio_str = pair.split(':')
        nc = int(nc_str.strip())
        ratio = parse_ratio(ratio_str)
        result[nc] = ratio
    return result


def parse_int_sep_ratios_by_Nc(s: str) -> Dict[int, List[tuple]]:
    """
    Parse int_sep_ratios_by_Nc dict from string.
    Format: 'Nc1:p1,q1|p2,q2;Nc2:p3,q3|p4,q4' -> {Nc1: [(p1,q1), (p2,q2)], ...}
    Example: '2:1,2|1,4|1,8;3:1,3|2,3' -> {2: [(1,2), (1,4), (1,8)], 3: [(1,3), (2,3)]}
    """
    result: Dict[int, List[tuple]] = {}
    for pair in s.split(';'):
        if ':' not in pair:
            continue
        nc_str, ratios_str = pair.split(':')
        nc = int(nc_str.strip())
        ratios = [parse_ratio(r) for r in ratios_str.split('|')]
        result[nc] = ratios
    return result


def run_filling_cluster_sizes(args):
    """Run compare_filling_cluster_sizes with parsed arguments."""
    from aah_code.cluster_model.plots import compare_filling_cluster_sizes

    # Parse int_sep_ratios
    if args.int_sep_ratios:
        int_sep_ratios = parse_int_sep_ratios(args.int_sep_ratios)
    else:
        # Default: maximal separation for each cluster size
        int_sep_ratios = {nc: (1, nc) for nc in args.cluster_sizes}

    print("=" * 60)
    print("Running compare_filling_cluster_sizes")
    print("=" * 60)
    print(f"L={args.L}, chi={args.chi}")
    print(f"cluster_sizes={args.cluster_sizes}")
    print(f"int_sep_ratios={int_sep_ratios}")
    print(f"U_values={args.U_values}")
    print(f"V_values={args.V_values}")
    print(f"v_sep_ratio={args.v_sep_ratio}")
    if args.results:
        print(f"Loading from: {args.results}")
    print("=" * 60)

    fig, results = compare_filling_cluster_sizes(
        v_sep_ratio=args.v_sep_ratio,
        int_sep_ratios=int_sep_ratios,
        cluster_sizes=args.cluster_sizes,
        U_values=args.U_values,
        V_values=args.V_values,
        L=args.L,
        t=args.t,
        chi=args.chi,
        solver_method=args.solver_method,
        states_retained=args.states_retained,
        include_idmrg=args.include_idmrg,
        include_finite_dmrg=args.include_finite_dmrg,
        save_data=args.save_data,
        save_html=args.save_html,
        show_plots=args.show_plots,
        plot_relative_error=args.plot_relative_error,
        results=args.results,
    )

    print("\nDone!")
    return fig, results


def run_filling_with_int_cluster_sizes(args):
    """Run compare_filling_with_int_cluster_sizes with parsed arguments."""
    from aah_code.cluster_model.plots import compare_filling_with_int_cluster_sizes

    # Parse int_sep_ratios_by_Nc
    if args.int_sep_ratios_by_Nc:
        int_sep_ratios_by_Nc = parse_int_sep_ratios_by_Nc(args.int_sep_ratios_by_Nc)
    else:
        raise ValueError("int_sep_ratios_by_Nc is required for this command")

    print("=" * 60)
    print("Running compare_filling_with_int_cluster_sizes")
    print("=" * 60)
    print(f"L={args.L}, chi={args.chi}")
    print(f"int_sep_ratios_by_Nc={int_sep_ratios_by_Nc}")
    print(f"U_values={args.U_values}")
    print(f"V={args.V}")
    print(f"v_sep_ratio={args.v_sep_ratio}")
    print(f"set_filling={args.set_filling}")
    if args.results:
        print(f"Loading from: {args.results}")
    print("=" * 60)

    fig, results = compare_filling_with_int_cluster_sizes(
        int_sep_ratios_by_Nc=int_sep_ratios_by_Nc,
        U_values=args.U_values,
        V=args.V,
        v_sep_ratio=args.v_sep_ratio,
        t=args.t,
        L=args.L,
        chi=args.chi,
        solver_method=args.solver_method,
        states_retained=args.states_retained,
        include_idmrg=args.include_idmrg,
        include_finite_dmrg=args.include_finite_dmrg,
        save_data=args.save_data,
        save_html=args.save_html,
        show_plots=args.show_plots,
        plot_relative_error=args.plot_relative_error,
        set_filling=args.set_filling,
        dmrg_fixed_filling=args.dmrg_fixed_filling,
        results=args.results,
        finite_dmrg_bc=args.finite_dmrg_bc,
    )

    print("\nDone!")
    return fig, results


def run_fixed_supercluster(args):
    """Run compare_fixed_supercluster with parsed arguments."""
    from aah_code.cluster_model.plots import compare_fixed_supercluster

    print("=" * 60)
    print("Running compare_fixed_supercluster")
    print("=" * 60)
    print(f"L={args.L}, chi={args.chi}")
    print(f"supercluster_sizes={args.supercluster_sizes}")
    print(f"max_seps={args.max_seps}")
    print(f"set_filling={args.set_filling}")
    print(f"U_values={args.U_values}")
    print(f"V_values={args.V_values}")
    print(f"v_sep_ratio={args.v_sep_ratio}")
    if args.results:
        print(f"Loading from: {args.results}")
    print("=" * 60)

    fig, results = compare_fixed_supercluster(
        v_sep_ratio=args.v_sep_ratio,
        U_values=args.U_values,
        V_values=args.V_values,
        supercluster_sizes=args.supercluster_sizes,
        max_seps=args.max_seps,
        set_filling=args.set_filling,
        L=args.L,
        t=args.t,
        chi=args.chi,
        solver_method=args.solver_method,
        states_retained=args.states_retained,
        include_idmrg=args.include_idmrg,
        include_finite_dmrg=args.include_finite_dmrg,
        save_data=args.save_data,
        save_html=args.save_html,
        show_plots=args.show_plots,
        plot_relative_error=args.plot_relative_error,
        log_yaxis=args.log_yaxis,
        axes=tuple(args.axes),
        rows_per_page=args.rows_per_page,
        cols_per_page=args.cols_per_page,
        results=args.results,
    )

    print("\nDone!")
    return fig, results


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    import yaml

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    # Apply variable substitution (e.g., ${L} -> actual L value)
    config = substitute_config_vars(config)

    return config


def substitute_config_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Substitute ${var} patterns in string values with actual config values.
    E.g., output_dir: "filling_L${L}" with L: 120 -> output_dir: "filling_L120"
    """
    import re

    def substitute_string(s: str, variables: Dict[str, Any]) -> str:
        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            if var_name in variables:
                return str(variables[var_name])
            return match.group(0)  # Leave unchanged if not found
        return re.sub(r'\$\{(\w+)\}', replacer, s)

    def substitute_recursive(obj: Any, variables: Dict[str, Any]) -> Any:
        if isinstance(obj, str):
            return substitute_string(obj, variables)
        elif isinstance(obj, dict):
            return {k: substitute_recursive(v, variables) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [substitute_recursive(item, variables) for item in obj]
        return obj

    return substitute_recursive(config, config)


def run_from_config(config: Dict[str, Any]):
    """
    Run comparison function from a config dict (loaded from YAML).

    Expected YAML structure:
    ```yaml
    command: filling_cluster_sizes  # or fixed_supercluster

    # Common parameters
    L: 48
    chi: 32
    t: 1.0
    v_sep_ratio: [1, 2]
    U_values: [0, 1, 2, 3, 4, 5, 7, 10, 20, 30]
    V_values: [1e-6, 1, 2, 3, 5]
    solver_method: sparse_ED
    states_retained: 6
    include_finite_dmrg: true
    include_idmrg: false
    save_data: true
    save_html: false
    show_plots: false
    plot_relative_error: false
    results: null  # or path to pickle file

    # For filling_cluster_sizes:
    cluster_sizes: [2, 4, 6, 8]
    int_sep_ratios:
      2: [1, 2]
      4: [1, 4]
      6: [1, 6]
      8: [1, 8]

    # For fixed_supercluster:
    supercluster_sizes: [2, 4, 6, 8]
    max_seps: 4
    set_filling: 1.0
    log_yaxis: false
    rows_per_page: 4
    cols_per_page: 4
    ```
    """
    command = config.get('command')
    if not command:
        raise ValueError("Config must specify 'command' (filling_cluster_sizes or fixed_supercluster)")

    # Convert v_sep_ratio from list to tuple if needed
    v_sep = config.get('v_sep_ratio', [1, 2])
    if isinstance(v_sep, list):
        v_sep = tuple(v_sep)

    # Convert int_sep_ratios dict values from lists to tuples
    int_sep_ratios = config.get('int_sep_ratios')
    if int_sep_ratios:
        int_sep_ratios = {int(k): tuple(v) for k, v in int_sep_ratios.items()}

    print("=" * 60)
    print(f"Running from config: {command}")
    print("=" * 60)
    for key, value in config.items():
        if key != 'command':
            print(f"  {key}: {value}")
    print("=" * 60)

    if command == 'filling_cluster_sizes':
        from aah_code.cluster_model.plots import compare_filling_cluster_sizes

        # Build int_sep_ratios if not provided
        cluster_sizes = config.get('cluster_sizes', [2, 4, 6, 8])
        if not int_sep_ratios:
            int_sep_ratios = {nc: (1, nc) for nc in cluster_sizes}

        # Handle output_dir - if specified, use partials subdirectory
        output_dir = config.get('output_dir')
        if output_dir:
            output_dir = f'large_files/partials/{output_dir}'
        else:
            output_dir = 'large_files/plots'

        fig, results = compare_filling_cluster_sizes(
            v_sep_ratio=v_sep,
            int_sep_ratios=int_sep_ratios,
            cluster_sizes=cluster_sizes,
            U_values=config.get('U_values', [0, 1, 2, 3, 4, 5, 7, 10, 20, 30]),
            V_values=config.get('V_values', [1e-6, 1, 2, 3, 5]),
            L=config.get('L', 24),
            t=config.get('t', 1.0),
            chi=config.get('chi', 32),
            solver_method=config.get('solver_method', 'sparse_ED'),
            states_retained=config.get('states_retained', 6),
            include_idmrg=config.get('include_idmrg', False),
            include_finite_dmrg=config.get('include_finite_dmrg', True),
            save_data=config.get('save_data', True),
            save_html=config.get('save_html', False),
            show_plots=config.get('show_plots', False),
            plot_relative_error=config.get('plot_relative_error', False),
            output_dir=output_dir,
            results=config.get('results'),
        )

    elif command == 'filling_with_int_cluster_sizes':
        from aah_code.cluster_model.plots import compare_filling_with_int_cluster_sizes

        # Convert int_sep_ratios_by_Nc from config format
        int_sep_by_nc_raw = config.get('int_sep_ratios_by_Nc')
        if not int_sep_by_nc_raw:
            raise ValueError("int_sep_ratios_by_Nc is required for filling_with_int_cluster_sizes")
        int_sep_ratios_by_Nc = {
            int(k): [tuple(r) for r in v]
            for k, v in int_sep_by_nc_raw.items()
        }

        # Handle output_dir - if specified, use partials subdirectory
        output_dir = config.get('output_dir')
        if output_dir:
            output_dir = f'large_files/partials/{output_dir}'
        else:
            output_dir = 'large_files/plots'

        fig, results = compare_filling_with_int_cluster_sizes(
            int_sep_ratios_by_Nc=int_sep_ratios_by_Nc,
            U_values=config.get('U_values', [0, 1, 2, 3, 4, 5, 7, 10, 20, 30]),
            V=config.get('V', 0.0),
            v_sep_ratio=v_sep,
            t=config.get('t', 1.0),
            L=config.get('L', 24),
            chi=config.get('chi', 32),
            solver_method=config.get('solver_method', 'sparse_ED'),
            states_retained=config.get('states_retained', 6),
            include_idmrg=config.get('include_idmrg', False),
            include_finite_dmrg=config.get('include_finite_dmrg', True),
            save_data=config.get('save_data', True),
            save_html=config.get('save_html', False),
            show_plots=config.get('show_plots', False),
            plot_relative_error=config.get('plot_relative_error', False),
            set_filling=config.get('set_filling'),
            dmrg_fixed_filling=config.get('dmrg_fixed_filling', True),
            include_timing=config.get('include_timing', False),
            output_dir=output_dir,
            results=config.get('results'),
        )

    elif command == 'fixed_supercluster':
        from aah_code.cluster_model.plots import compare_fixed_supercluster

        # Handle output_dir - if specified, use partials subdirectory
        output_dir = config.get('output_dir')
        if output_dir:
            output_dir = f'large_files/partials/{output_dir}'
        else:
            output_dir = 'large_files/plots'

        fig, results = compare_fixed_supercluster(
            v_sep_ratio=v_sep,
            U_values=config.get('U_values', [0, 1, 2, 3, 4, 5, 7, 10, 20, 30]),
            V_values=config.get('V_values', [1e-6, 1, 2, 3, 5]),
            supercluster_sizes=config.get('supercluster_sizes', [2, 4, 6, 8]),
            max_seps=config.get('max_seps', 4),
            set_filling=config.get('set_filling', 1.0),
            L=config.get('L', 24),
            t=config.get('t', 1.0),
            chi=config.get('chi', 32),
            solver_method=config.get('solver_method', 'sparse_ED'),
            states_retained=config.get('states_retained', 6),
            include_idmrg=config.get('include_idmrg', False),
            include_finite_dmrg=config.get('include_finite_dmrg', True),
            save_data=config.get('save_data', True),
            save_html=config.get('save_html', False),
            show_plots=config.get('show_plots', False),
            plot_relative_error=config.get('plot_relative_error', False),
            log_yaxis=config.get('log_yaxis', False),
            axes=tuple(config.get('axes', ['U', 'V'])),
            rows_per_page=config.get('rows_per_page', 4),
            cols_per_page=config.get('cols_per_page', 4),
            output_dir=output_dir,
            results=config.get('results'),
        )
    else:
        raise ValueError(f"Unknown command: {command}")

    print("\nDone!")
    return fig, results


def main():
    # Check for --config argument first (before full argparse)
    if len(sys.argv) >= 2 and sys.argv[1].startswith('--config'):
        if '=' in sys.argv[1]:
            config_path = sys.argv[1].split('=', 1)[1]
        elif len(sys.argv) >= 3:
            config_path = sys.argv[2]
        else:
            print("Error: --config requires a path argument")
            sys.exit(1)

        config = load_yaml_config(config_path)
        run_from_config(config)
        return

    parser = argparse.ArgumentParser(
        description="Run cluster comparison functions on the cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    subparsers = parser.add_subparsers(dest='command', help='Comparison function to run')

    # Common arguments for all commands
    def add_common_args(p):
        p.add_argument('--L', type=int, default=24, help='System size')
        p.add_argument('--chi', type=int, default=32, help='DMRG bond dimension')
        p.add_argument('--t', type=float, default=1.0, help='Hopping parameter')
        p.add_argument('--solver_method', type=str, default='sparse_ED',
                       choices=['dense_ED', 'sparse_ED'], help='Solver method')
        p.add_argument('--states_retained', type=int, default=6, help='States retained')
        p.add_argument('--v_sep_ratio', type=parse_ratio, default=(1, 2),
                       help='V separation ratio as p,q (default: 1,2 for π)')
        p.add_argument('--U_values', type=parse_float_list, default=[0, 1, 2, 3, 4, 5, 7, 10, 20, 30],
                       help='Comma-separated U values')
        p.add_argument('--V_values', type=parse_float_list, default=[1e-6, 1, 2, 3, 5],
                       help='Comma-separated V values')
        p.add_argument('--include_idmrg', action='store_true', help='Include iDMRG')
        p.add_argument('--include_finite_dmrg', action='store_true', default=True,
                       help='Include finite DMRG')
        p.add_argument('--no_finite_dmrg', action='store_false', dest='include_finite_dmrg',
                       help='Disable finite DMRG')
        p.add_argument('--save_data', action='store_true', default=True, help='Save pickle')
        p.add_argument('--no_save_data', action='store_false', dest='save_data')
        p.add_argument('--save_html', action='store_true', help='Save HTML plot')
        p.add_argument('--show_plots', action='store_true', help='Show plots (disabled on cluster)')
        p.add_argument('--plot_relative_error', action='store_true', help='Plot relative error')
        p.add_argument('--results', type=str, default=None,
                       help='Path to existing results pickle to load instead of computing')

    # filling_cluster_sizes subcommand
    fcs_parser = subparsers.add_parser('filling_cluster_sizes',
                                        help='Compare filling across cluster sizes')
    add_common_args(fcs_parser)
    fcs_parser.add_argument('--cluster_sizes', type=parse_int_list, default=[2, 4, 6, 8],
                            help='Comma-separated cluster sizes')
    fcs_parser.add_argument('--int_sep_ratios', type=str, default=None,
                            help='Int sep ratios as "Nc1:p1,q1;Nc2:p2,q2" (default: 1/Nc for each)')

    # filling_with_int_cluster_sizes subcommand
    fwi_parser = subparsers.add_parser('filling_with_int_cluster_sizes',
                                        help='Compare filling across cluster sizes with multiple int_seps per Nc')
    add_common_args(fwi_parser)
    fwi_parser.add_argument('--int_sep_ratios_by_Nc', type=str, required=True,
                            help='Int sep ratios as "Nc1:p1,q1|p2,q2;Nc2:p3,q3" (required)')
    fwi_parser.add_argument('--V', type=float, default=0.0,
                            help='Single V value (default: 0.0)')
    fwi_parser.add_argument('--set_filling', type=float, default=None,
                            help='Target filling (default: None for half/quarter comparison)')
    fwi_parser.add_argument('--dmrg_fixed_filling', action='store_true', default=True,
                            help='Use fixed particle number in finite DMRG')
    fwi_parser.add_argument('--no_dmrg_fixed_filling', action='store_false', dest='dmrg_fixed_filling',
                            help='Use grand canonical DMRG')
    fwi_parser.add_argument('--finite_dmrg_bc', type=str, default='open', choices=['open', 'periodic'],
                            help='Boundary conditions for finite DMRG (default: open)')

    # fixed_supercluster subcommand
    fsc_parser = subparsers.add_parser('fixed_supercluster',
                                        help='Compare fixed supercluster configurations')
    add_common_args(fsc_parser)
    fsc_parser.add_argument('--supercluster_sizes', type=parse_int_list, default=[2, 4, 6, 8],
                            help='Comma-separated supercluster sizes')
    fsc_parser.add_argument('--max_seps', type=int, default=4,
                            help='Max (Nc, int_sep) pairs per supercluster')
    fsc_parser.add_argument('--set_filling', type=float, default=1.0,
                            help='Target filling (1.0=half, 0.5=quarter)')
    fsc_parser.add_argument('--log_yaxis', action='store_true', help='Log scale y-axis')
    fsc_parser.add_argument('--rows_per_page', type=int, default=4, help='Rows per page')
    fsc_parser.add_argument('--cols_per_page', type=int, default=4, help='Columns per page')
    fsc_parser.add_argument('--axes', nargs=2, default=['U', 'V'],
                            help='What goes on x-axis vs rows: U V (default) or V U')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == 'filling_cluster_sizes':
        run_filling_cluster_sizes(args)
    elif args.command == 'filling_with_int_cluster_sizes':
        run_filling_with_int_cluster_sizes(args)
    elif args.command == 'fixed_supercluster':
        run_fixed_supercluster(args)
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == '__main__':
    main()
