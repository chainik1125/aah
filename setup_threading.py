#!/usr/bin/env python3
"""
Helper script to detect CPU environment and set optimal threading configuration
for NumPy, SciPy, and other numerical libraries.
"""

import os
import multiprocessing
import subprocess
import sys

def get_cpu_info():
    """Get detailed CPU information."""
    cpu_count = multiprocessing.cpu_count()
    
    # Try to get more detailed info from lscpu
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        lscpu_output = result.stdout
        
        # Parse relevant info
        cores_per_socket = 1
        sockets = 1
        threads_per_core = 1
        
        for line in lscpu_output.split('\n'):
            if 'Core(s) per socket:' in line:
                cores_per_socket = int(line.split(':')[1].strip())
            elif 'Socket(s):' in line:
                sockets = int(line.split(':')[1].strip())
            elif 'Thread(s) per core:' in line:
                threads_per_core = int(line.split(':')[1].strip())
        
        physical_cores = cores_per_socket * sockets
        
        return {
            'cpu_count': cpu_count,
            'physical_cores': physical_cores,
            'threads_per_core': threads_per_core,
            'sockets': sockets
        }
    except:
        # Fallback if lscpu not available
        return {
            'cpu_count': cpu_count,
            'physical_cores': cpu_count,
            'threads_per_core': 1,
            'sockets': 1
        }

def detect_and_set_threads(set_env=True, verbose=True):
    """
    Detect CPU configuration and optionally set thread environment variables.
    
    Args:
        set_env: If True, set environment variables. If False, just return recommendations.
        verbose: If True, print information about the configuration.
    
    Returns:
        dict: Recommended thread settings
    """
    cpu_info = get_cpu_info()
    
    # Determine if this is a many-CPU environment
    is_many_cpu = cpu_info['cpu_count'] >= 8
    
    # Recommended thread count - use physical cores for compute-heavy tasks
    # to avoid hyperthreading overhead for numerical operations
    if is_many_cpu:
        # For large systems, use physical cores or a reasonable maximum
        recommended_threads = min(cpu_info['physical_cores'], 64)
    else:
        # For smaller systems, use all available
        recommended_threads = cpu_info['cpu_count']
    
    thread_settings = {
        'OMP_NUM_THREADS': str(recommended_threads),
        'OPENBLAS_NUM_THREADS': str(recommended_threads),
        'MKL_NUM_THREADS': str(recommended_threads),
        'NUMEXPR_NUM_THREADS': str(recommended_threads),
        'VECLIB_MAXIMUM_THREADS': str(recommended_threads),
        'NUMBA_NUM_THREADS': str(recommended_threads),
    }
    
    if verbose:
        print("=" * 60)
        print("CPU CONFIGURATION DETECTION")
        print("=" * 60)
        print(f"Total CPU cores available: {cpu_info['cpu_count']}")
        print(f"Physical cores: {cpu_info['physical_cores']}")
        print(f"Threads per core: {cpu_info['threads_per_core']}")
        print(f"Sockets: {cpu_info['sockets']}")
        print(f"Many-CPU environment: {'Yes' if is_many_cpu else 'No'}")
        print(f"\nRecommended thread count: {recommended_threads}")
        print("(Using physical cores for optimal numerical performance)")
    
    if set_env:
        # Check current settings
        current_settings = {}
        for var in thread_settings:
            current = os.environ.get(var, 'not set')
            current_settings[var] = current
        
        if verbose:
            print("\n" + "-" * 60)
            print("CURRENT THREAD SETTINGS:")
            print("-" * 60)
            for var, value in current_settings.items():
                print(f"{var}: {value}")
        
        # Set the environment variables
        for var, value in thread_settings.items():
            os.environ[var] = value
        
        if verbose:
            print("\n" + "-" * 60)
            print("NEW THREAD SETTINGS:")
            print("-" * 60)
            for var, value in thread_settings.items():
                print(f"{var}: {value}")
            
            print("\n" + "=" * 60)
            print("✓ Thread environment variables have been set!")
            print("=" * 60)
    else:
        if verbose:
            print("\n" + "-" * 60)
            print("RECOMMENDED SETTINGS (not applied):")
            print("-" * 60)
            for var, value in thread_settings.items():
                print(f"export {var}={value}")
    
    return thread_settings

def create_env_file(filename='.env.threading'):
    """Create a shell script file with the threading environment variables."""
    settings = detect_and_set_threads(set_env=False, verbose=False)
    
    with open(filename, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Threading environment variables for optimal performance\n")
        f.write("# Source this file before running Python scripts:\n")
        f.write("#   source .env.threading\n\n")
        
        for var, value in settings.items():
            f.write(f"export {var}={value}\n")
    
    print(f"Environment file created: {filename}")
    print(f"To use: source {filename}")
    
    return filename

def benchmark_threading():
    """Run a simple benchmark to test threading performance."""
    import numpy as np
    import time
    
    print("\n" + "=" * 60)
    print("THREADING BENCHMARK")
    print("=" * 60)
    
    # Matrix sizes to test
    sizes = [1000, 2000, 3000]
    
    for size in sizes:
        print(f"\nMatrix multiplication ({size}x{size}):")
        
        # Create random matrices
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        
        # Time the multiplication
        start = time.perf_counter()
        C = A @ B
        elapsed = time.perf_counter() - start
        
        print(f"  Time: {elapsed:.3f} seconds")
        print(f"  GFLOPS: {2 * size**3 / elapsed / 1e9:.2f}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Configure threading for numerical libraries")
    parser.add_argument('--no-set', action='store_true', 
                       help="Don't set environment variables, just show recommendations")
    parser.add_argument('--quiet', action='store_true',
                       help="Suppress verbose output")
    parser.add_argument('--benchmark', action='store_true',
                       help="Run a benchmark after setting threads")
    parser.add_argument('--create-env-file', action='store_true',
                       help="Create a .env.threading file for sourcing")
    parser.add_argument('--threads', type=int,
                       help="Manually specify number of threads to use")
    
    args = parser.parse_args()
    
    if args.create_env_file:
        create_env_file()
    else:
        # Detect and set threads
        settings = detect_and_set_threads(
            set_env=not args.no_set,
            verbose=not args.quiet
        )
        
        # Override with manual thread count if specified
        if args.threads and not args.no_set:
            thread_count = str(args.threads)
            for var in settings:
                os.environ[var] = thread_count
            if not args.quiet:
                print(f"\nManually overridden to use {args.threads} threads")
        
        # Run benchmark if requested
        if args.benchmark:
            # Only run benchmark if we actually set the variables
            if not args.no_set:
                benchmark_threading()
            else:
                print("\nSkipping benchmark (environment variables not set)")